from dotenv import load_dotenv

from llm.abstract_llm import AbstractLlm

load_dotenv()



class GeminiLlm(AbstractLlm):
    def __init__(self):
        pass
    
    def generate(self, prompt: str) -> str:
        return "Gemini response for: " + prompt