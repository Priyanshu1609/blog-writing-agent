# Abstract class to generate llm

from abc import ABC, abstractmethod

class AbstractLlm(ABC):
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    