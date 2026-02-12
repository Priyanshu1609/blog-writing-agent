from __future__ import annotations

from typing import Any, Mapping, cast

from langchain_core.messages import SystemMessage, HumanMessage

from llm.impl.openai import OpenAILlm
from models.schemas import State, Task, Plan, EvidenceItem
from prompts.worker_prompt import WORKER_SYSTEM
from steps.abstract_step import AbstractStep


class WorkerStep(AbstractStep):
	@classmethod
	def execute(cls, state: State) -> dict:
		payload = cast(Mapping[str, Any], state)
		task = Task(**payload["task"])
		plan = Plan(**payload["plan"])
		evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

		bullets_text = "\n- " + "\n- ".join(task.bullets)
		evidence_text = "\n".join(
			f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
			for e in evidence[:20]
		)

		llm = OpenAILlm.get_llm()
		response = llm.invoke(
			[
				SystemMessage(content=WORKER_SYSTEM),
				HumanMessage(
					content=(
						f"Blog title: {plan.blog_title}\n"
						f"Audience: {plan.audience}\n"
						f"Tone: {plan.tone}\n"
						f"Blog kind: {plan.blog_kind}\n"
						f"Constraints: {plan.constraints}\n"
						f"Topic: {payload['topic']}\n"
						f"Mode: {payload.get('mode')}\n"
						f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
						f"Section title: {task.title}\n"
						f"Goal: {task.goal}\n"
						f"Target words: {task.target_words}\n"
						f"Tags: {task.tags}\n"
						f"requires_research: {task.requires_research}\n"
						f"requires_citations: {task.requires_citations}\n"
						f"requires_code: {task.requires_code}\n"
						f"Bullets:{bullets_text}\n\n"
						f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
					)
				),
			]
		)

		content = response.content
		if isinstance(content, list):
			content = "\n".join(str(item) for item in content)
		section_md = str(content).strip()

		return {"sections": [(task.id, section_md)]}
