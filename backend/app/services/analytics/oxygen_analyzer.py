from typing import Dict
from typing import List


class OxygenAnalyzer:

    def analyze(
        self,
        rows: List[dict]
    ) -> Dict:

        valid_rows = [

            row

            for row in rows

            if row.get(
                "oxygen_umol_kg"
            ) is not None
        ]

        if len(valid_rows) < 2:

            return {

                "success": False,

                "error":
                    "Insufficient oxygen data"
            }

        valid_rows.sort(
            key=lambda x: x["depth_m"]
        )

        surface_oxygen = (
            valid_rows[0]
            ["oxygen_umol_kg"]
        )

        deep_oxygen = (
            valid_rows[-1]
            ["oxygen_umol_kg"]
        )

        min_row = min(

            valid_rows,

            key=lambda x:
                x["oxygen_umol_kg"]
        )

        max_row = max(

            valid_rows,

            key=lambda x:
                x["oxygen_umol_kg"]
        )
        depths = [

            row["depth_m"]

            for row in valid_rows
        ]

        oxygen_values = [

            row["oxygen_umol_kg"]

            for row in valid_rows
        ]

        return {

            "success": True,

            "analysis_type":
                "oxygen_analysis",

            "surface_oxygen":
                round(
                    surface_oxygen,
                    2
                ),

            "deep_oxygen":
                round(
                    deep_oxygen,
                    2
                ),

            "minimum_oxygen":
                round(
                    min_row[
                        "oxygen_umol_kg"
                    ],
                    2
                ),

            "minimum_depth":
                round(
                    min_row[
                        "depth_m"
                    ],
                    2
                ),

            "maximum_oxygen":
                round(
                    max_row[
                        "oxygen_umol_kg"
                    ],
                    2
                ),

            "oxygen_change":
                round(
                    deep_oxygen
                    -
                    surface_oxygen,
                    2
                ),

            "confidence":
                "high"
                if len(valid_rows) > 50
                else "medium",
            "depths":
                depths,
            "oxygen":
                oxygen_values
        }