from app.services.analytics.mixed_layer_depth_analyzer import (
    MixedLayerDepthAnalyzer
)

from app.services.presentation.profile_visualization_builder import (
    ProfileVisualizationBuilder
)


class MixedLayerDepthHandler:

    def __init__(self):

        self.analyzer = (
            MixedLayerDepthAnalyzer()
        )

        self.visualizer = (
            ProfileVisualizationBuilder()
        )

    async def process(
        self,
        rows
    ):

        analysis = (
            self.analyzer.analyze(
                rows
            )
        )

        visualization = (
            self.visualizer
            .build_temperature_profile(

                rows,

                analysis.get(
                    "mld_depth"
                )
            )
        )

        return {

            "analysis":
                analysis,

            "visualization":
                visualization
        }