from sentence_transformers import (
    SentenceTransformer
)


class EmbeddingService:

    def __init__(self):

        self.model = (
            SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
        )

    def embed_texts(
        self,
        texts
    ):

        return self.model.encode(
            texts
        ).tolist()

    def embed_query(
        self,
        query
    ):

        return self.model.encode(
            query
        ).tolist()