# from sentence_transformers import (
#     SentenceTransformer
# )
import google.generativeai as genai
from app.core.config import config

class EmbeddingService:

    def __init__(self):

        # self.model = (
        #     SentenceTransformer(
        #         "all-MiniLM-L6-v2"
        #     )
        # )
        import os
        api_key = config.GOOGLE_API_KEY or os.getenv("GEMINI_KEY_1") or os.getenv("GEMINI_KEY_2")
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = "models/embedding-001" # Using stable Gemini embedding model

    def embed_texts(
        self,
        texts
    ):
        
        # return self.model.encode(
        #     texts
        # ).tolist()
        result = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type="retrieval_document"
        )
        # Returns a dict where 'embedding' is a list of lists if multiple texts
        return result['embedding']

    def embed_query(
        self,
        query
    ):

        # return self.model.encode(
        #     query
        # ).tolist()
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']