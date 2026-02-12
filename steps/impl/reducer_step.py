from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import cast

from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from llm.impl.openai import OpenAILlm
from models.schemas import State, GlobalImagePlan
from prompts.decide_images_prompt import DECIDE_IMAGES_SYSTEM
from steps.abstract_step import AbstractStep


class MergeContentStep(AbstractStep):
    @classmethod
    def execute(cls, state: State) -> dict:
        plan = state["plan"]
        if plan is None:
            raise ValueError("merge_content called without plan.")
        ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
        body = "\n\n".join(ordered_sections).strip()
        merged_md = f"# {plan.blog_title}\n\n{body}\n"
        return {"merged_md": merged_md}


class DecideImagesStep(AbstractStep):
    @classmethod
    def execute(cls, state: State) -> dict:
        llm = OpenAILlm.get_llm()
        planner = llm.with_structured_output(GlobalImagePlan)
        merged_md = state["merged_md"]
        plan = state["plan"]
        assert plan is not None

        image_plan = cast(
            GlobalImagePlan,
            planner.invoke(
                [
                    SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Blog kind: {plan.blog_kind}\n"
                            f"Topic: {state['topic']}\n\n"
                            "Insert placeholders + propose image prompts.\n\n"
                            f"{merged_md}"
                        )
                    ),
                ]
            ),
        )

        return {
            "md_with_placeholders": image_plan.md_with_placeholders,
            "image_specs": [img.model_dump() for img in image_plan.images],
        }


class GenerateAndPlaceImagesStep(AbstractStep):
    @staticmethod
    def _gemini_generate_image_bytes(prompt: str) -> bytes:
        from google import genai
        from google.genai import types

        api_key = settings.google_api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")

        client = genai.Client(api_key=api_key)

        resp = client.models.generate_content(
            model=settings.image_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    )
                ],
            ),
        )

        parts = getattr(resp, "parts", None)
        candidates = getattr(resp, "candidates", None)
        if not parts and candidates:
            try:
                parts = candidates[0].content.parts
            except Exception:
                parts = None

        if not parts:
            raise RuntimeError("No image content returned (safety/quota/SDK change).")

        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data

        raise RuntimeError("No inline image bytes found in response.")

    @staticmethod
    def _safe_slug(title: str) -> str:
        value = title.strip().lower()
        value = re.sub(r"[^a-z0-9 _-]+", "", value)
        value = re.sub(r"\s+", "_", value).strip("_")
        return value or "blog"

    @staticmethod
    def _output_paths(state: State, blog_title: str) -> tuple[Path, Path]:
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(exist_ok=True)
        images_dir = output_dir / settings.images_dir
        images_dir.mkdir(exist_ok=True)

        suffix = ""
        if state.get("mode") == "open_book":
            as_of = state.get("as_of") or date.today().isoformat()
            suffix = f"_{as_of}"

        filename = f"{GenerateAndPlaceImagesStep._safe_slug(blog_title)}{suffix}.md"
        return output_dir / filename, images_dir

    @classmethod
    def execute(cls, state: State) -> dict:
        plan = state["plan"]
        assert plan is not None

        md = state.get("md_with_placeholders") or state["merged_md"]
        image_specs = state.get("image_specs", []) or []

        output_path, images_dir = cls._output_paths(state, plan.blog_title)

        if not image_specs:
            output_path.write_text(md, encoding="utf-8")
            return {"final": md, "final_path": str(output_path), "blog_title": plan.blog_title}

        for spec in image_specs:
            placeholder = spec["placeholder"]
            filename = spec["filename"]
            out_path = images_dir / filename

            if not out_path.exists():
                try:
                    img_bytes = cls._gemini_generate_image_bytes(spec["prompt"])
                    out_path.write_bytes(img_bytes)
                except Exception as exc:
                    prompt_block = (
                        f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                        f"> **Alt:** {spec.get('alt','')}\n>\n"
                        f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                        f"> **Error:** {exc}\n"
                    )
                    md = md.replace(placeholder, prompt_block)
                    continue

            img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
            md = md.replace(placeholder, img_md)

        output_path.write_text(md, encoding="utf-8")
        return {"final": md, "final_path": str(output_path), "blog_title": plan.blog_title}
