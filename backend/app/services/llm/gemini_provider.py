import os
import asyncio
import google.generativeai as genai
from pathlib import Path

from dotenv import load_dotenv
env_path = Path(".env")
load_dotenv(env_path)

class GeminiProvider:

    def __init__(self):

        self.keys = [

            os.getenv("GEMINI_KEY_1"),
            os.getenv("GEMINI_KEY_2"),
            os.getenv("GEMINI_KEY_3"),
            os.getenv("GEMINI_KEY_4"),
            os.getenv("GEMINI_KEY_5"),
            os.getenv("GEMINI_KEY_6"),
        ]

        self.keys = [

            key

            for key in self.keys

            if key
        ]

        self.current_index = 0
        if not self.keys:
            raise Exception(
                "No Gemini API keys found"
            )

    async def generate(
        self,
        prompt: str
    ):

        last_error = None

        for _ in range(len(self.keys)):

            api_key = self.keys[self.current_index]

            self.current_index = (
                self.current_index + 1
            ) % len(self.keys)

            try:

                genai.configure(
                    api_key=api_key
                )

                model = genai.GenerativeModel(
                    "gemini-2.5-flash"
                )

                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt
                )

                # DEBUG
                print("RAW RESPONSE:")
                print(response)

                if not response.candidates:
                    raise Exception(
                        "No candidates returned"
                    )

                candidate = response.candidates[0]

                print(
                    "FINISH REASON:",
                    candidate.finish_reason
                )

                if (
                    not hasattr(candidate, "content")
                    or not candidate.content
                    or not candidate.content.parts
                ):
                    raise Exception(
                        f"No content returned. Finish reason: {candidate.finish_reason}"
                    )

                text = ""

                for part in candidate.content.parts:

                    if hasattr(part, "text"):

                        text += part.text

                if not text.strip():

                    raise Exception(
                        "Gemini returned empty text"
                    )

                return text.strip()

            except Exception as e:

                print(
                    f"Gemini key failed: {e}"
                )

                last_error = e

                continue

        raise Exception(
            f"All Gemini keys failed: {last_error}"
        )