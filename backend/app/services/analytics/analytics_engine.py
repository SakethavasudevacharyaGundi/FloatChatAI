from app.services.analytics.trend_analysis import (
    TrendAnalysisService
)

from app.services.analytics.comparision_service import (
    ComparisonService
)

from app.services.analytics.anamoly_detector import (
    AnomalyDetector
)

from app.services.analytics.water_column_analyzer import (
    WaterColumnAnalyzer
)
from app.services.analytics.mixed_layer_depth_analyzer import (
    MixedLayerDepthAnalyzer
)
from app.services.analytics.salinity_analyzer import (
    SalinityAnalyzer
)

from app.services.analytics.oxygen_analyzer import (
    OxygenAnalyzer
)

from app.services.analytics.omz_analyzer import (
    OMZDetector
)

from app.services.analytics.trajectory_analyzer import (
    TrajectoryAnalyzer
)
import logging 


class AnalyticsEngine:

    def __init__(self):

        self.trend = (
            TrendAnalysisService()
        )

        self.comparison = (
            ComparisonService()
        )

        self.anomaly = (
            AnomalyDetector()
        )

        self.water_column = (
            WaterColumnAnalyzer()
        )
        self.mld = (
            MixedLayerDepthAnalyzer()
        )
        self.salinity = (
        SalinityAnalyzer()
        )

        self.oxygen = (
            OxygenAnalyzer()
        )

        self.omz = (
            OMZDetector()
        )

        self.trajectory = (
            TrajectoryAnalyzer()
        )

    def run(
        self,
        intent_type,
        rows
    ):
        logger = logging.getLogger(__name__)
        logger.info(
    f"INTENT TYPE RECEIVED: {intent_type}"
)


        if intent_type == "trend_analysis":
            logger.info("TREND ROW SAMPLE:")
            logger.info(rows[:5])
            return (
                self.trend
                .analyze_temperature_trend(
                    rows
                )
            )

        elif intent_type == "anomaly_detection":
            
            return (
                self.anomaly.detect(
                    rows
                )
            )
        elif intent_type == "mixed_layer_depth":

            return (
                self.mld.analyze(
                    rows
                )
            )
        elif intent_type == "salinity_analysis":

            return (
                self.salinity.analyze(
                    rows
                )
            )

        elif intent_type == "oxygen_profile":
            return (
                self.oxygen.analyze(
                    rows
                )
            )

        elif intent_type == "oxygen_analysis":

            return (
                self.oxygen.analyze(
                    rows
                )
            )

        elif intent_type == "omz_detection":

            return (
                self.omz.analyze(
                    rows
                )
            )

        elif intent_type == "trajectory_analysis":

            return (
                self.trajectory.analyze(
                    rows
                )
            )
        elif intent_type == "water_column":

            return (
                self.water_column
                .analyze(
                    rows
                )
            )

        elif intent_type == "float_comparison":
            return (
                self.comparison
                .compare_from_rows(
                    rows
                )
            )

        return {}