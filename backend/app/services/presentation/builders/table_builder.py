class TableBuilder:

    def build(
        self,
        analysis
    ):

        return {

            "type": "table",

            "columns": [

                "metric",

                "float_a",

                "float_b"
            ],

            "rows": [

                [

                    "Average Temperature",

                    analysis.get(
                        "avg_temperature_a"
                    ),

                    analysis.get(
                        "avg_temperature_b"
                    )
                ],

                [

                    "Average Salinity",

                    analysis.get(
                        "avg_salinity_a"
                    ),

                    analysis.get(
                        "avg_salinity_b"
                    )
                ],

                [

                    "Maximum Depth",

                    analysis.get(
                        "max_depth_a"
                    ),

                    analysis.get(
                        "max_depth_b"
                    )
                ],

                [

                    "Profile Count",

                    analysis.get(
                        "profile_count_a"
                    ),

                    analysis.get(
                        "profile_count_b"
                    )
                ]
            ]
        }