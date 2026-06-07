import hashlib
from app.services.sql.sql_generation_service import (
    SQLGenerationService
)

from app.database.query_executor import (
    QueryExecutor
)

from app.services.ai.summary_service import (
    SummaryService
)

from app.services.sql.sql_intent_parser import (
    SQLIntentParser
)

from app.services.analytics.analytics_engine import (
    AnalyticsEngine
)

from app.services.sql.sql_validator import (
    SQLValidator
)
from app.services.presentation.presentation_generator import (
    PresentationGenerator
)
from app.cache.cache_service import (
    CacheService
)

class AnalyticalHandler:

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

        self.analytics_engine = (
            AnalyticsEngine()
        )

        self.intent_parser = (
            SQLIntentParser()
        )
        self.presentation_generator = (
            PresentationGenerator()
        )

    async def process(
        self,
        query: str
    ):

        intent = (
            self.intent_parser.parse(
                query
            )
        )
        print("INTENT PARSED:")
        print(intent)

        sql = (
            await self.sql_generator
            .generate_sql(query, intent["type"])
        )

        SQLValidator.validate(sql)
        cache_key = (

            "sql:" +

            hashlib.md5(
                sql.encode()
            ).hexdigest()
        )
        cached_rows = CacheService.get(
            cache_key
        )

        if cached_rows:

            rows = cached_rows

        else:

            rows = self.executor.execute(
                sql
            )

            CacheService.set(
                cache_key,
                rows,
                ttl=1800
            )
        print("ROW COUNT:")
        print(len(rows))

        if rows:
            print("FIRST ROW:")
            print(rows[0])

        analysis = (
            self.analytics_engine.run(
                intent["type"],
                rows
            )
        )
        print(f"ANALYSIS RESULT: {analysis}")
        presentation = (
            self.presentation_generator
            .generate(
                intent["type"],
                analysis
            )
        )
        summary_analysis = {

            k: v

            for k, v in analysis.items()

            if k not in [

                "depths",
                "temperatures",
                "salinity",
                "oxygen",
                "latitudes",
                "longitudes",
                "timestamps"
            ]
        }
        summary = (
            await self.summary_service
            .summarize_analysis(
                intent["type"],
                summary_analysis
            )
        )

        return {

            "response":
                summary,

            "analysis":
                analysis,
            
            "presentation":                
                presentation,

            "generated_sql":
                sql,

            "rows_returned":
                len(rows)
        }