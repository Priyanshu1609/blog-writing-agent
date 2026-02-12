"""Abstract class for each step in the graph."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.schemas import State


class AbstractStep(ABC):
    @classmethod
    @abstractmethod
    def execute(cls, state: State) -> dict:
        raise NotImplementedError