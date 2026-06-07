from typing import Dict
from typing import List


class OMZDetector:

    OMZ_THRESHOLD = 150

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

        if not valid_rows:

            return {

                "success": False,

                "error":
                    "No oxygen data"
            }

        omz_rows = [

            row

            for row in valid_rows

            if row[
                "oxygen_umol_kg"
            ] < self.OMZ_THRESHOLD
        ]

        if not omz_rows:

            return {

                "success": True,

                "analysis_type":
                    "omz_detection",

                "omz_detected":
                    False,

                "confidence":
                    "high"
            }

        return {

            "success": True,

            "analysis_type":
                "omz_detection",

            "omz_detected":
                True,

            "start_depth":
                omz_rows[0]["depth_m"],

            "end_depth":
                omz_rows[-1]["depth_m"],

            "minimum_oxygen":
                min(
                    row[
                        "oxygen_umol_kg"
                    ]
                    for row
                    in omz_rows
                ),

            "points_in_omz":
                len(omz_rows),

            "confidence":
                "high"
        }