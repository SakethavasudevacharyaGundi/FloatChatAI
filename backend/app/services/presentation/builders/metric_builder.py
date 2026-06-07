class MetricBuilder:

    def build(
        self,
        analysis
    ):

        return {

            "type":
                "metric_card",

            "metrics":

                analysis
        }