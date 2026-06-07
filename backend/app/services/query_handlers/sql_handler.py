from app.services.sql.sql_intent_parser import SQLIntentParser
from app.services.sql.sql_template_builder import SQLTemplateBuilder
from app.database.query_executor import QueryExecutor
from app.services.analytics.statistics_service import StatisticsService
from app.services.ai.summary_service import SummaryService

from app.database.repositories.measurement_repository import (
    MeasurementRepository
)

from app.database.repositories.profile_repository import (
    ProfileRepository
)
from app.database.repositories.float_repository import (
    FloatRepository
)
from app.services.analytics.trend_analysis import (
    TrendAnalysisService
)



class SQLHandler:

    def __init__(self):

        self.intent_parser = SQLIntentParser()
        self.template_builder = SQLTemplateBuilder()
        self.executor = QueryExecutor()

        self.statistics_service = StatisticsService()
        self.summary_service = SummaryService()

        self.measurement_repository = MeasurementRepository()
        self.profile_repository = ProfileRepository()
        self.float_repository = FloatRepository()
        self.trend_analysis = TrendAnalysisService()
        


    async def process(self, query: str):

        try:

            intent = self.intent_parser.parse(query)

            print(f"INTENT: {intent}")

            if intent["type"] == "latest_profile":

                profile = (
                    self.profile_repository
                    .get_latest_profile(
                        intent.get("float_id")
                    )
                )

                return {
                    "response":
                        f"Latest profile found for float {intent.get('float_id')}",
                    "profile": profile
                }


            elif intent["type"] == "temperature_profile":

                rows = (
                    self.measurement_repository
                    .get_temperature_profile(
                        intent.get("float_id")
                    )
                )

                stats = (
                    self.statistics_service
                    .summarize_temperature_profile(rows)
                )

                summary = (
                    await self.summary_service
                    .summarize_temperature_trend(
                        intent.get("float_id"),
                        stats
                    )
                )

                print(f"ROWS RETURNED: {len(rows)}")

                return {
                    "response": summary,
                    "statistics": stats,
                    "data_preview": rows[:50]
                }
            elif intent["type"] == "float_summary":

                summary = (
                    self.float_repository
                    .get_float_summary(
                        intent.get("float_id")
                    )
                )

                return {
                    "response":
                        f"Summary for float {intent.get('float_id')}",
                    "float": summary
                }
            elif intent["type"] == "trend_analysis":

                rows = (self.measurement_repository.get_temperature_trend(intent.get("float_id")));

                trend_data = (
                    self.trend_analysis
                    .analyze_temperature_trend(
                        rows
                    )
                )

                trend_summary = (
                    await self.summary_service
                    .summarize_temperature_trend(
                        intent.get("float_id"),
                        trend_data
                    )
                )
                timeser = (self.measurement_repository.get_temperature_profile(intent.get("float_id")));

                return {
                    "response": trend_summary,
                    "trend_data": trend_data,
                    "timeser" : timeser
                }
            elif intent["type"] == "float_comparison":

                float_a_rows = (
                    self.measurement_repository
                    .get_temperature_profile(
                        intent["float_a"]
                    )
                )

                float_b_rows = (
                    self.measurement_repository
                    .get_temperature_profile(
                        intent["float_b"]
                    )
                )

                profiles_a = (
                    self.profile_repository
                    .get_profiles_by_float(
                        intent["float_a"]
                    )
                )

                profiles_b = (
                    self.profile_repository
                    .get_profiles_by_float(
                        intent["float_b"]
                    )
                )

                comparison = (
                    self.comparison_service.compare(
                        float_a_rows,
                        float_b_rows,
                        profiles_a,
                        profiles_b
                    )
                )

                summary = (
                    await self.summary_service
                    .summarize_float_comparison(
                        intent["float_a"],
                        intent["float_b"],
                        comparison
                    )
                )

                return {

                    "response": summary,

                    "comparison": comparison
                }
            return {
                "response":
                    f"Unsupported query type: {intent['type']}",
                "data": []
            }

        except Exception as e:

            print(f"SQL Handler Error: {e}")

            return {
                "response":
                    f"SQL execution failed: {str(e)}",
                "data": []
            }