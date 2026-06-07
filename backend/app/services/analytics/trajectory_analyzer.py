from math import radians
from math import sin
from math import cos
from math import sqrt
from math import atan2


class TrajectoryAnalyzer:

    def _haversine(
        self,
        lat1,
        lon1,
        lat2,
        lon2
    ):

        R = 6371

        dlat = radians(
            lat2 - lat1
        )

        dlon = radians(
            lon2 - lon1
        )

        a = (

            sin(dlat / 2) ** 2

            +

            cos(
                radians(lat1)
            )

            *

            cos(
                radians(lat2)
            )

            *

            sin(dlon / 2) ** 2
        )

        return (
            2
            *
            R
            *
            atan2(
                sqrt(a),
                sqrt(1 - a)
            )
        )

    def analyze(
        self,
        rows
    ):

        if len(rows) < 2:

            return {

                "success": False,

                "error":
                    "Insufficient trajectory data"
            }

        rows.sort(

            key=lambda x:
                x["profile_datetime"]
        )

        total_distance = 0

        for i in range(
            1,
            len(rows)
        ):

            total_distance += (
                self._haversine(

                    rows[i - 1]["lat"],
                    rows[i - 1]["lon"],

                    rows[i]["lat"],
                    rows[i]["lon"]
                )
            )
        latitudes = [

            row["lat"]

            for row in rows
        ]

        longitudes = [

            row["lon"]

            for row in rows
        ]

        timestamps = [

            str(
                row["profile_datetime"]
            )

            for row in rows
        ]

        return {

            "success": True,

            "analysis_type":
                "trajectory_analysis",

            "profiles":
                len(rows),

            "distance_km":
                round(
                    total_distance,
                    2
                ),

            "start_lat":
                rows[0]["lat"],

            "start_lon":
                rows[0]["lon"],

            "end_lat":
                rows[-1]["lat"],

            "end_lon":
                rows[-1]["lon"],

            "confidence":
                "high",
            "latitudes":
                latitudes,
            "longitudes":
                longitudes,
            "timestamps":
                timestamps
        }