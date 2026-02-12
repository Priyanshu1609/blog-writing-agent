from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1-mini"
    openai_temperature: float = 0.2

    tavily_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    image_model: str = "gemini-2.5-flash-image"

    output_dir: str = "outputs"
    images_dir: str = "images"

    default_mode: str = "closed_book"
    default_recency_days: int = 3650
    default_topic: str = "Latest tech news"
    default_as_of: Optional[str] = None

    blog_branch: str = "blogs"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
