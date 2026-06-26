from app.services.sql.sql_generation_service import (
    SQLGenerationService
)

from app.database.query_executor import (
    QueryExecutor
)

from app.services.ai.summary_service import (
    SummaryService
)
from app.services.sql.sql_validator import (
    SQLValidator
)
from app.cache.sql_cache import ( 
    SQLCache
    )


class SQLGenerationHandler:

    def __init__(self):

        self.sql_generator = (
            SQLGenerationService()
        )

        self.executor = (
            QueryExecutor()
        )

        self.summary_service = (
            SummaryService()
        )


    async def process(
        self,
        query: str
    ):

        try:

            sql = (
                await self.sql_generator
                .generate_sql(query)
            )

            print("GENERATED SQL:")
            print(sql)
            SQLValidator.validate(sql)

            rows = SQLCache.get(sql)

            if rows:

                print(
                    "SQL CACHE HIT"
                )

            else:

                print(
                    "SQL CACHE MISS"
                )

                rows = (
                    self.executor.execute(sql)
                )

                SQLCache.set(
                    sql,
                    rows
                )
            print("EXECUTOR RETURNED:")
            print(rows)

            print("EXECUTOR ROW COUNT:")
            print(len(rows))

            summary = (
                await self.summary_service
                .summarize_generated_results(
                    query,
                    rows[:20]
                )
            )

            return {

                "response":
                    summary,

                "generated_sql":
                    sql,

                "rows_returned":
                    len(rows),

                "data":
                    rows[:50]
            }

        except Exception as e:

            print(
                f"SQL Generation Error: {e}"
            )

            return {

                "response":
                    f"Failed: {str(e)}",

                "data": []
            }