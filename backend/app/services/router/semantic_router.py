# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
from app.services.router.capability_registry import (
    CapabilityRegistry
)
from app.services.vector.embedding_service import EmbeddingService

import numpy as np

def cosine_similarity(a, b):
    # Simple cosine similarity implementation using numpy to avoid sklearn dependency
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class SemanticRouter:

    def __init__(self):

        # self.model = SentenceTransformer(
        #     "all-MiniLM-L6-v2"
        # )
        self.embedding_service = EmbeddingService()

        self.capabilities = (
            CapabilityRegistry
            .get_capabilities()
        )

        self.capability_embeddings = {}

        self._build_embeddings()

    def _build_embeddings(self):

        for route_name, route_info in (
            self.capabilities.items()
        ):

            examples = route_info["examples"]

            # embeddings = self.model.encode(
            #     examples
            # )
            embeddings = self.embedding_service.embed_texts(examples)

            avg_embedding = np.mean(
                embeddings,
                axis=0
            )

            self.capability_embeddings[
                route_name
            ] = avg_embedding

    def route(self, query: str):

        # query_embedding = self.model.encode(
        #     query
        # )
        query_embedding = self.embedding_service.embed_query(query)

        best_route = None
        best_score = -1
        matched_example = None

        for route_name, route_info in (
            self.capabilities.items()
        ):

            route_embedding = (
                self.capability_embeddings[
                    route_name
                ]
            )

            # score = cosine_similarity(
            #     [query_embedding],
            #     [route_embedding]
            # )[0][0]
            score = cosine_similarity(query_embedding, route_embedding)
            
            print(
                    f"{route_name}: {score}"
                )

            if score > best_score:

                best_score = score
                best_route = route_name

                matched_example = (
                    route_info["examples"][0]
                )

        threshold = (
            self.capabilities[
                best_route
            ]["threshold"]
        )

        if best_score >= threshold:

            return {
                "route": best_route,
                "handler":
                    self.capabilities[
                        best_route
                    ]["handler"],
                "score":
                    float(best_score),
                "threshold":
                    threshold,
                "matched_example":
                    matched_example
            }

        return {
            "route": "general",
            "handler": "general_handler",
            "score": float(best_score),
            "threshold": threshold,
            "matched_example": None
        }