class ProfileVisualizationBuilder:

    def build_temperature_profile(
        self,
        rows,
        mld_depth=None
    ):

        return {

            "type":
                "temperature_profile",

            "depths": [

                row["depth_m"]

                for row in rows
            ],

            "temperatures": [

                row["temperature_c"]

                for row in rows
            ],

            "markers": {

                "mixed_layer_depth":
                    mld_depth
            }
        }