from app.services.presentation.presentation_registry import (
    PresentationRegistry
)

from app.services.presentation.builders.chart_builder import (
    ChartBuilder
)

from app.services.presentation.builders.table_builder import (
    TableBuilder
)

from app.services.presentation.builders.metric_builder import (
    MetricBuilder
)

from app.services.presentation.builders.map_builder import (
    MapBuilder
)


class PresentationGenerator:

    def __init__(self):

        self.chart_builder = (
            ChartBuilder()
        )

        self.table_builder = (
            TableBuilder()
        )

        self.metric_builder = (
            MetricBuilder()
        )

        self.map_builder = (
            MapBuilder()
        )

    def generate(
        self,
        analysis_type,
        analysis
    ):

        presentation_type = (
            PresentationRegistry
            .PRESENTATIONS
            .get(
                analysis_type
            )
        )

        if presentation_type == "chart":

            return (
                self.chart_builder
                .build(
                    analysis_type,
                    analysis
                )
            )

        elif presentation_type == "table":

            return (
                self.table_builder
                .build(
                    analysis
                )
            )

        elif presentation_type == "metric_card":

            return (
                self.metric_builder
                .build(
                    analysis
                )
            )

        elif presentation_type == "map":

            return (
                self.map_builder
                .build(
                    analysis
                )
            )

        return None