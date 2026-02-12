from __future__ import annotations

from datetime import date
from typing import List, cast

from langchain_core.messages import SystemMessage, HumanMessage

from llm.impl.openai import OpenAILlm
from models.schemas import State, RouterDecision
from prompts.router_prompt import ROUTER_SYSTEM
from steps.abstract_step import AbstractStep


class RouterStep(AbstractStep):
    @staticmethod
    def _default_recency(mode: str) -> int:
        if mode == "open_book":
            return 7
        if mode == "hybrid":
            return 45
        return 3650

    @staticmethod
    def _normalize_queries(topic: str, queries: List[str] | None) -> List[str]:
        if queries:
            return [q.strip() for q in queries if q.strip()]
        return [topic]

    @classmethod
    def execute(cls, state: State) -> dict:
        forced_mode = state.get("mode")
        forced_queries = state.get("queries")
        as_of = state.get("as_of") or date.today().isoformat()

        if forced_mode in {"closed_book", "hybrid", "open_book"}:
            mode = forced_mode
            queries = cls._normalize_queries(state["topic"], forced_queries)
            needs_research = mode != "closed_book"
            recency_days = state.get("recency_days") or cls._default_recency(mode)
            return {
                "needs_research": needs_research,
                "mode": mode,
                "queries": queries if needs_research else [],
                "recency_days": recency_days,
                "as_of": as_of,
            }

        llm = OpenAILlm.get_llm()
        decider = llm.with_structured_output(RouterDecision)
        decision = cast(
            RouterDecision,
            decider.invoke(
                [
                    SystemMessage(content=ROUTER_SYSTEM),
                    HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {as_of}"),
                ]
            ),
        )

        recency_days = state.get("recency_days") or cls._default_recency(decision.mode)

        return {
            "needs_research": decision.needs_research,
            "mode": decision.mode,
            "queries": cls._normalize_queries(state["topic"], decision.queries),
            "recency_days": recency_days,
            "as_of": as_of,
        }

    @classmethod
    def route_next(cls, state: State) -> str:
        return "research" if state["needs_research"] else "orchestrator"