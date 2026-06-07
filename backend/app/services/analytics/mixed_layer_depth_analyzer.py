from typing import Dict
from typing import List
from typing import Optional


class MixedLayerDepthAnalyzer:

    DEFAULT_TEMP_THRESHOLD = 0.2

    def analyze(
        self,
        rows: List[dict],
        threshold: float = DEFAULT_TEMP_THRESHOLD
    ) -> Dict:

        if not rows:

            return {
                "success": False,
                "error": "No profile data available"
            }

        valid_rows = [

            row

            for row in rows

            if row.get("depth_m") is not None
            and row.get("temperature_c") is not None
        ]

        if len(valid_rows) < 2:

            return {
                "success": False,
                "error": (
                    "Insufficient temperature profile"
                )
            }

        valid_rows.sort(
            key=lambda x: x["depth_m"]
        )

        surface_temp = (
            valid_rows[0]["temperature_c"]
        )

        target_temp = (
            surface_temp - threshold
        )

        mld_depth: Optional[float] = None

        for row in valid_rows:

            if (
                row["temperature_c"]
                <= target_temp
            ):

                mld_depth = (
                    row["depth_m"]
                )

                break
        depths = [

            row["depth_m"]

            for row in valid_rows
        ]

        temperatures = [

            row["temperature_c"]

            for row in valid_rows
        ]

        return {

            "success": True,

            "analysis_type":
                "mixed_layer_depth",

            "surface_temperature":
                round(surface_temp, 3),

            "threshold_temperature":
                round(target_temp, 3),

            "mld_depth":
                mld_depth,

            "profile_points":
                len(valid_rows),

            "confidence":
                self._calculate_confidence(
                    valid_rows,
                    mld_depth
                ),
            "depths":

                depths,
            "temperatures":
                temperatures
        }

    def _calculate_confidence(
        self,
        rows: List[dict],
        mld_depth: Optional[float]
    ) -> str:

        if mld_depth is None:

            return "low"

        if len(rows) > 100:

            return "high"

        if len(rows) > 30:

            return "medium"

        return "low"