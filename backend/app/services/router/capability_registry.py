class CapabilityRegistry:

    @staticmethod
    def get_capabilities():

        return {

            # =========================
            # RETRIEVAL ROUTES
            # =========================

            "temperature_profile": {

                "handler": "sql",

                "threshold": 0.69,

                "examples": [

                    "show temperature profile",

                    "show temperature measurements",

                    "temperature data for float",

                    "temperature readings",

                    "show temperature observations",

                    "temperature profile for float"

                ]
            },

            "latest_profile": {

                "handler": "sql",

                "threshold": 0.69,

                "examples": [

                    "latest profile",

                    "show latest profile",

                    "show me latest profile",

                    "latest observation",

                    "most recent profile",

                    "newest profile",

                    "newest observation",

                    "last profile",

                    "most recent measurement",

                    "last reported data",

                    "latest data from float",

                    "show last observation",

                    "show latest observation",

                    "when did this float last report",

                    "what is the newest profile"

                ]
            },
            "mixed_layer_depth": {

                "handler": "analytical",

                "threshold": 0.60,

                "examples": [

                    "calculate mixed layer depth",
                    "find mixed layer depth",
                    "estimate mld",
                    "determine mixed layer depth",
                    "mixed layer analysis",
                    "what is the mixed layer depth",
                    "surface layer depth",
                    "analyze mixed layer"
                ]
            },

            "salinity_analysis": {

                "handler": "analytical",

                "threshold": 0.60,

                "examples": [

                    "analyze salinity profile",
                    "salinity analysis",
                    "salinity structure",
                    "salinity distribution",
                    "how does salinity vary with depth",
                    "show salinity structure"
                ]
            },

            "oxygen_analysis": {

                "handler": "analytical",

                "threshold": 0.60,

                "examples": [

                    "analyze oxygen profile",
                    "oxygen analysis",
                    "oxygen structure",
                    "oxygen distribution",
                    "show oxygen profile",
                    "how does oxygen vary with depth"
                ]
            },

            "omz_detection": {

                "handler": "analytical",

                "threshold": 0.60,

                "examples": [

                    "detect omz",
                    "oxygen minimum zone",
                    "find oxygen minimum zone",
                    "identify omz",
                    "low oxygen region"
                ]
            },

            "trajectory_analysis": {

                "handler": "analytical",

                "threshold": 0.60,

                "examples": [

                    "analyze float trajectory",
                    "show float trajectory",
                    "float path",
                    "float movement",
                    "track float",
                    "drift path"
                ]
            },

            "float_summary": {

                "handler": "sql",

                "threshold": 0.69,

                "examples": [

                    "float summary",

                    "overview of float",

                    "information about float",

                    "float details",

                    "describe this float",

                    "tell me about this float"

                ]
            },

            # =========================
            # ANALYTICAL ROUTES
            # =========================

            "trend_analysis": {

                "handler": "analytical",

                "threshold": 0.69,

                "examples": [

                    "temperature trend",

                    "warming trend",

                    "analyze temperature changes",

                    "has this float been warming",

                    "show warming trends",

                    "has the float been warming",

                    "analyze temperature trend",

                    "temperature changes over time",

                    "cooling trend",

                    "temperature evolution"

                ]
            },

            "float_comparison": {

                "handler": "analytical",

                "threshold": 0.69,

                "examples": [

                    "compare floats",

                    "compare two floats",

                    "compare float",

                    "comparison between floats",

                    "compare float 1900975 and 1900979",

                    "which float is warmer",

                    "which float is deeper",

                    "compare temperature between floats",

                    "compare profiles",

                    "float comparison"

                ]
            },
            "ocean_knowledge": {
                "handler": "vector",
                "threshold": 0.60,
                "examples": [

                    "what is a thermocline",

                    "define thermocline",

                    "explain thermocline",

                    "what is ocean stratification",

                    "define stratification",

                    "what is a pycnocline",

                    "what is mixed layer depth",

                    "what is an argo float",

                    "how does an argo float work",

                    "what causes ocean warming",

                    "why does oxygen decrease with depth",

                    "what is dissolved oxygen",

                    "what are ocean currents",

                    "what is a water column"
                ]
            },

            "anomaly_detection": {

                "handler": "analytical",

                "threshold": 0.69,

                "examples": [

                    "find anomalies",

                    "detect anomalies",

                    "detect unusual temperatures",

                    "find outliers",

                    "show unusual observations",

                    "temperature anomalies",

                    "abnormal measurements",

                    "unusual ocean conditions",

                    "identify outliers",

                    "anomaly detection"

                ]
            },

            "water_column": {

                "handler": "analytical",

                "threshold": 0.69,

                "examples": [

                    "water column analysis",

                    "analyze water column",

                    "vertical ocean structure",

                    "water column structure",

                    "thermocline depth",

                    "show thermocline",

                    "analyze ocean layers",

                    "vertical temperature structure",

                    "ocean stratification",

                    "describe water column"

                ]
            }

        }