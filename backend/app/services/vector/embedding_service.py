# from sentence_transformers import (
#     SentenceTransformer
# )
import google.generativeai as genai
from app.core.config import config

class EmbeddingService:

    def __init__(self):
        import os
        self.keys = [
            os.getenv("GEMINI_KEY_1"),
            os.getenv("GEMINI_KEY_2"),
            os.getenv("GEMINI_KEY_3"),
            os.getenv("GEMINI_KEY_4"),
            os.getenv("GEMINI_KEY_5"),
            os.getenv("GEMINI_KEY_6"),
            config.GOOGLE_API_KEY
        ]
        self.keys = [key for key in self.keys if key]
        self.current_index = 0
        if not self.keys:
            print("WARNING: No Gemini API keys found!")
            
        self.model_name = "models/gemini-embedding-2"

    def _execute_with_rotation(self, content, task_type):
        last_error = None
        for _ in range(len(self.keys)):
            api_key = self.keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.keys)
            try:
                genai.configure(api_key=api_key)
                result = genai.embed_content(
                    model=self.model_name,
                    content=content,
                    task_type=task_type
                )
                return result['embedding']
            except Exception as e:
                print(f"Embedding key failed: {e}")
                last_error = e
                continue
        raise Exception(f"All Gemini keys failed during embedding: {last_error}")

    def embed_texts(self, texts):
        return self._execute_with_rotation(texts, "retrieval_document")

    def embed_query(self, query):
        return self._execute_with_rotation(query, "retrieval_query")
