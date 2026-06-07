import requests

from app.services.llm.base_provider import BaseProvider


class OllamaProvider(BaseProvider):

    def __init__(self):
        self.model = "qwen2.5:7b"

    async def generate(self, prompt: str):

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,

                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
        )

        response.raise_for_status()

        return response.json()["response"]