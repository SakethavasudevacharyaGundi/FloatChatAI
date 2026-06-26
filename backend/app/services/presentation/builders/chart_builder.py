class ChartBuilder:

    def build(
        self,
        analysis_type,
        analysis
    ):

        if analysis_type == "trend_analysis":

            return {

                "type": "chart",

                "chart_type": "line",

                "title":
                    "Temperature Trend",

                "x":
                    analysis.get(
                        "dates",
                        []
                    ),

                "y":
                    analysis.get(
                        "temperatures",
                        []
                    )
            }

        elif analysis_type == "water_column":

            return {

                "type": "chart",

                "chart_type": "profile",

                "title":
                    "Temperature Profile",

                "depths":
                    analysis.get(
                        "depths",
                        []
                    ),

                "temperatures":
                    analysis.get(
                        "temperatures",
                        []
                    )
            }

        elif analysis_type == "mixed_layer_depth":

            return {

                "type": "chart",

                "chart_type": "profile",

                "title":
                    "Mixed Layer Depth",

                "depths":
                    analysis.get(
                        "depths",
                        []
                    ),

                "temperatures":
                    analysis.get(
                        "temperatures",
                        []
                    ),

                "mld":
                    analysis.get(
                        "mixed_layer_depth"
                    )
            }

        elif analysis_type == "salinity_analysis":

            return {

                "type": "chart",

                "chart_type": "profile",

                "title":
                    "Salinity Profile",

                "depths":
                    analysis.get(
                        "depths",
                        []
                    ),

                "salinity":
                    analysis.get(
                        "salinity",
                        []
                    )
            }

        elif analysis_type == "oxygen_analysis":

            return {

                "type": "chart",

                "chart_type": "profile",

                "title":
                    "Oxygen Profile",

                "depths":
                    analysis.get(
                        "depths",
                        []
                    ),

                "oxygen":
                    analysis.get(
                        "oxygen",
                        []
                    )
            }

        return None