from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from config import settings
from llm.abstract_llm import AbstractLlm

load_dotenv()


class OpenAILlm(AbstractLlm):
    _client: Optional[ChatOpenAI] = None

    def __init__(self):
        pass

    @classmethod
    def get_llm(cls) -> ChatOpenAI:
        if cls._client is None:
            if settings.openai_api_key:
                os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
            cls._client = ChatOpenAI(
                model=settings.openai_model,
                temperature=settings.openai_temperature,
            )
        return cls._client

    def generate(self, prompt: str) -> str:
        content = self.get_llm().invoke(prompt).content
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)