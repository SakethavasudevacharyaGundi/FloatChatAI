from app.services.vector.embedding_service import (
    EmbeddingService
)

from app.services.vector.vector_store import (
    VectorStore
)


class VectorRetriever:

    def __init__(self):

        self.embedder = (
            EmbeddingService()
        )

        self.store = (
            VectorStore()
        )

    def retrieve(
        self,
        query,
        top_k=5
    ):

        query_embedding = (
            self.embedder
            .embed_query(query)
        )

        results = (
            self.store.search(
                query_embedding,
                top_k
            )
        )

        return results