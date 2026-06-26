class MapBuilder:

    def build(
        self,
        analysis
    ):

        return {

            "type":
                "map",

            "latitudes":
                analysis.get(
                    "latitudes",
                    []
                ),

            "longitudes":
                analysis.get(
                    "longitudes",
                    []
                ),

            "timestamps":
                analysis.get(
                    "timestamps",
                    []
                )
        }