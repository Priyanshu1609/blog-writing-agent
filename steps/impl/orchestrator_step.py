from __future__ import annotations

from typing import cast

from langchain_core.messages import SystemMessage, HumanMessage

from llm.impl.openai import OpenAILlm
from models.schemas import State, Plan
from prompts.orchestrator_prompt import ORCH_SYSTEM
from steps.abstract_step import AbstractStep


class OrchestratorStep(AbstractStep):
    @classmethod
    def execute(cls, state: State) -> dict:
        llm = OpenAILlm.get_llm()
        planner = llm.with_structured_output(Plan)
        mode = state.get("mode", "closed_book")
        evidence = state.get("evidence", [])

        forced_kind = "news_roundup" if mode == "open_book" else None

        plan = cast(
            Plan,
            planner.invoke(
                [
                    SystemMessage(content=ORCH_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Topic: {state['topic']}\n"
                            f"Mode: {mode}\n"
                            f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                            f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                            f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                        )
                    ),
                ]
            ),
        )
        if forced_kind:
            plan.blog_kind = "news_roundup"

        return {"plan": plan}
