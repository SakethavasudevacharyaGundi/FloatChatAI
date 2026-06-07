from app.services.query_handlers.sql_generation_handler import (
    SQLGenerationHandler
)
from app.services.ai.summary_service import (
    SummaryService
)


class GeneralHandler:

    def __init__(self):

        self.sql_generation_handler = (
            SQLGenerationHandler()
        )

        self.summary_service = (
            SummaryService()
        )

    async def process(
        self,
        query: str
    ):

        result = await (
            self.sql_generation_handler
            .process(query)
        )

        rows = result.get(
            "data",
            []
        )
        print("ROWS:")
        print(rows[:5])

        print("ROW COUNT:")
        print(len(rows))

        summary = await (
            self.summary_service
            .summarize_generated_results(
                query=query,
                rows=rows[:20]
            )
        )

        return {

            "response": summary,

            "data": rows
        }