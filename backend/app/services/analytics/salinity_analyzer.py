from typing import Dict
from typing import List


class SalinityAnalyzer:

    def analyze(
        self,
        rows: List[dict]
    ) -> Dict:

        valid_rows = [

            row

            for row in rows

            if row.get(
                "salinity_psu"
            ) is not None

            and row.get(
                "depth_m"
            ) is not None
        ]

        if len(valid_rows) < 2:

            return {

                "success": False,

                "error":
                    "Insufficient salinity data"
            }

        valid_rows.sort(
            key=lambda x: x["depth_m"]
        )

        surface_salinity = (
            valid_rows[0]["salinity_psu"]
        )

        deep_salinity = (
            valid_rows[-1]["salinity_psu"]
        )

        salinity_range = abs(
            deep_salinity
            - surface_salinity
        )

        max_gradient = 0
        halocline_depth = None

        for i in range(
            1,
            len(valid_rows)
        ):

            dz = (
                valid_rows[i]["depth_m"]
                -
                valid_rows[i - 1]["depth_m"]
            )

            if dz <= 0:
                continue

            gradient = abs(

                (
                    valid_rows[i]["salinity_psu"]
                    -
                    valid_rows[i - 1]["salinity_psu"]
                ) / dz
            )

            if gradient > max_gradient:

                max_gradient = gradient

                halocline_depth = (
                    valid_rows[i]["depth_m"]
                )
            depths = [

                row["depth_m"]

                for row in valid_rows
            ]

            salinity_values = [

                row["salinity_psu"]

                for row in valid_rows
            ]

        return {

            "success": True,

            "analysis_type":
                "salinity_analysis",

            "surface_salinity":
                round(
                    surface_salinity,
                    3
                ),

            "deep_salinity":
                round(
                    deep_salinity,
                    3
                ),

            "salinity_range":
                round(
                    salinity_range,
                    3
                ),

            "halocline_depth":
                halocline_depth,

            "max_gradient":
                round(
                    max_gradient,
                    5
                ),

            "confidence":
                "high"
                if len(valid_rows) > 50
                else "medium",
            "depths":
                depths,

            "salinity":
                salinity_values
        }