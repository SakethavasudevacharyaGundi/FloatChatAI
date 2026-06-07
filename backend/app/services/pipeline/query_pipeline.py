from app.services.ai.response_formatter import ResponseFormatter,ResponseStatus
from app.services.presentation.builders.chart_generator import ChartGenerator
from app.services.router.query_router import QueryRouter
from app.services.query_handlers.general_handler import GeneralHandler
from app.services.query_handlers.vector_handler import VectorHandler
from app.services.query_handlers.sql_handler import SQLHandler
from app.services.router.semantic_router import (
    SemanticRouter
)
from app.services.query_handlers.sql_generation_handler import (
    SQLGenerationHandler
)
from app.services.query_handlers.analytical_handlers import (
    AnalyticalHandler
)
from app.services.security.input_validator import (
    InputValidator
)


class QueryPipeline:

    def __init__(self):


        self.response_formatter = ResponseFormatter()

        self.chart_generator = ChartGenerator()

        self.router = SemanticRouter()

        self.general_handler = GeneralHandler()
        self.vector_handler = VectorHandler()
        self.sql_handler = SQLHandler()
        self.sql_generation_handler = SQLGenerationHandler()
        self.analytical_handler = AnalyticalHandler()
        self.input_validator = InputValidator()

    async def process(self, query: str):

        try:
            InputValidator.validate(query)

            routing_result = self.router.route(
                query
            )

            print(
                f"Routing Result => {routing_result}"
            )

            handler = routing_result["handler"]

            # =========================
            # SQL ROUTES
            # =========================

            if handler == "sql":

                return await (
                    self.sql_handler
                    .process(query)
                )

            # =========================
            # VECTOR ROUTES
            # =========================

            elif handler == "vector":
                
                return await (
                    self.vector_handler
                    .process(query)
                )

            # =========================
            # LLM SQL FALLBACK
            # =========================
            elif handler == "analytical":

                return await (
                    self.analytical_handler
                    .process(query)
                )
            elif handler == "sql_generation":

                return await (
                    self.sql_generation_handler
                    .process(query)
                )

            # =========================
            # GENERAL
            # =========================

            else:

                return await (
                    self.general_handler
                    .process(query)
                )

        except Exception as e:

            print(
                f"Pipeline error: {e}"
            )

            return {

                "status": "error",

                "query": query,

                "response":
                    f"Error processing query: {str(e)}",

                "data": []
            }