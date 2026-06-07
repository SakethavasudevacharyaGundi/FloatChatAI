from app.services.vector.vector_retriever import (
    VectorRetriever
)

from app.services.vector.context_builder import (
    ContextBuilder
)

from app.services.llm.gemini_provider import (
    GeminiProvider
)
from app.cache.vector_cache import (
    VectorCache
)


class VectorHandler:

    def __init__(self):

        self.retriever = (
            VectorRetriever()
        )

        self.context_builder = (
            ContextBuilder()
        )

        self.llm = (
            GeminiProvider()
        )

    async def process(
        self,
        query: str
    ):

        retrieval_results = (
            VectorCache.get(query)
        )

        if retrieval_results:

            print(
                "VECTOR CACHE HIT"
            )

        else:

            print(
                "VECTOR CACHE MISS"
            )

            retrieval_results = (
                self.retriever.retrieve(
                    query,
                    top_k=15
                )
            )

            VectorCache.set(
                query,
                retrieval_results
            )
        print("RAW RETRIEVAL RESULTS")
        print(retrieval_results)

        context_data = (
            self.context_builder.build(
                retrieval_results
            )
        )

        prompt = f"""
            You are an oceanography expert.

            Answer using only the provided context.
Rules:

1. Answer the user's question directly.

2. Do not discuss the dataset,
   SQL query,
   retrieved rows,
   database structure,
   missing data,
   or data availability.

3. Do not say:
   - "the dataset does not contain"
   - "the provided data does not show"
   - "based on the available data"
   - "the context does not contain"

4. Focus on providing the most useful
   scientific answer possible.

5. If the question cannot be answered from
   the information provided, answer briefly
   and naturally without discussing system
   limitations.

            Context:

            {context_data["context"]}

            Question:

            {query}
            """

        answer = (
            await self.llm.generate(
                prompt
            )
        )

        return {

            "response":
                answer,

            "sources":
                context_data[
                    "sources"
                ],
            "confidence":
            context_data[
                "confidence"
            ]
        }