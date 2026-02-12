from __future__ import annotations

import argparse
from datetime import date
from typing import cast

from config import settings
from graph import app
from models.schemas import State


def _build_state(args: argparse.Namespace) -> dict:
    topic = args.topic or settings.default_topic
    as_of = args.as_of or settings.default_as_of or date.today().isoformat()

    state: dict = {
        "topic": topic,
        "as_of": as_of,
    }

    if args.news:
        state["mode"] = "open_book"
        state["queries"] = [topic]

    if args.mode:
        state["mode"] = args.mode
    if args.recency_days:
        state["recency_days"] = args.recency_days
    if args.queries:
        state["queries"] = args.queries

    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Blog agent")
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["closed_book", "hybrid", "open_book"], default=None)
    parser.add_argument("--recency-days", type=int, default=None)
    parser.add_argument("--as-of", type=str, default=None)
    parser.add_argument("--queries", nargs="*", default=None)
    parser.add_argument("--news", action="store_true")

    args = parser.parse_args()
    state = _build_state(args)
    result = app.invoke(cast(State, state))
    output_path = result.get("final_path")
    if output_path:
        print(output_path)


if __name__ == "__main__":
    main()
