from __future__ import annotations

import os
from datetime import date, timedelta
from typing import List, Optional, cast

from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from llm.impl.openai import OpenAILlm
from models.schemas import State, EvidencePack
from prompts.research_prompt import RESEARCH_SYSTEM
from steps.abstract_step import AbstractStep


class ResearchStep(AbstractStep):
    @staticmethod
    def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
        if settings.tavily_api_key:
            os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)
        if not os.getenv("TAVILY_API_KEY"):
            return []
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore

            tool = TavilySearchResults(max_results=max_results)
            results = tool.invoke({"query": query})
            out: List[dict] = []
            for r in results or []:
                out.append(
                    {
                        "title": r.get("title") or "",
                        "url": r.get("url") or "",
                        "snippet": r.get("content") or r.get("snippet") or "",
                        "published_at": r.get("published_date") or r.get("published_at"),
                        "source": r.get("source"),
                    }
                )
            return out
        except Exception:
            return []

    @staticmethod
    def _iso_to_date(value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        try:
            return date.fromisoformat(value[:10])
        except Exception:
            return None

    @classmethod
    def execute(cls, state: State) -> dict:
        queries = (state.get("queries") or [])[:10]
        raw: List[dict] = []
        for q in queries:
            raw.extend(cls._tavily_search(q, max_results=6))

        if not raw:
            return {"evidence": []}

        llm = OpenAILlm.get_llm()
        extractor = llm.with_structured_output(EvidencePack)
        pack = cast(
            EvidencePack,
            extractor.invoke(
                [
                    SystemMessage(content=RESEARCH_SYSTEM),
                    HumanMessage(
                        content=(
                            f"As-of date: {state['as_of']}\n"
                            f"Recency days: {state['recency_days']}\n\n"
                            f"Raw results:\n{raw}"
                        )
                    ),
                ]
            ),
        )

        dedup = {}
        for e in pack.evidence:
            if e.url:
                dedup[e.url] = e
        evidence = list(dedup.values())

        if state.get("mode") == "open_book":
            as_of = date.fromisoformat(state["as_of"])
            cutoff = as_of - timedelta(days=int(state["recency_days"]))
            evidence = [
                e for e in evidence if (d := cls._iso_to_date(e.published_at)) and d >= cutoff
            ]

        return {"evidence": evidence}
