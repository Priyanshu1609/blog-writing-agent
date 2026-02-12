from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional, cast

from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import settings
from graph import app
from models.schemas import State


Mode = Literal["closed_book", "hybrid", "open_book"]


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=3)
    mode: Optional[Mode] = None
    recency_days: Optional[int] = Field(None, ge=1, le=3650)
    as_of: Optional[str] = None
    queries: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    path: str
    content: str
    mode: str
    blog_title: str


class GenerateNewsRequest(BaseModel):
    topic: Optional[str] = None
    queries: Optional[List[str]] = None
    as_of: Optional[str] = None


app_api = FastAPI(title="Blog Agent")


@app_api.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app_api.post("/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest) -> GenerateResponse:
    as_of = payload.as_of or settings.default_as_of or date.today().isoformat()

    state: dict = {
        "topic": payload.topic,
        "as_of": as_of,
    }
    if payload.mode:
        state["mode"] = payload.mode
    if payload.recency_days:
        state["recency_days"] = payload.recency_days
    if payload.queries:
        state["queries"] = payload.queries

    result = app.invoke(cast(State, state))
    return GenerateResponse(
        path=result.get("final_path", ""),
        content=result.get("final", ""),
        mode=state.get("mode") or result.get("mode", ""),
        blog_title=result.get("blog_title", ""),
    )


@app_api.post("/generate-news", response_model=GenerateResponse)
async def generate_news(payload: GenerateNewsRequest) -> GenerateResponse:
    topic = payload.topic or settings.default_topic
    as_of = payload.as_of or settings.default_as_of or date.today().isoformat()

    state: dict = {
        "topic": topic,
        "mode": "open_book",
        "queries": payload.queries or [topic],
        "as_of": as_of,
    }

    result = app.invoke(cast(State, state))
    return GenerateResponse(
        path=result.get("final_path", ""),
        content=result.get("final", ""),
        mode="open_book",
        blog_title=result.get("blog_title", ""),
    )
