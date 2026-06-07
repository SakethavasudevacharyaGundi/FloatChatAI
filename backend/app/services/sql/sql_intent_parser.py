import re


class SQLIntentParser:

    def parse(
        self,
        query: str
    ):

        q = query.lower()

        float_ids = re.findall(
            r"\b\d{7}\b",
            query
        )

        float_id = (
            float_ids[0]
            if float_ids
            else None
        )

        # =========================
        # ANALYTICAL INTENTS
        # =========================

# =========================
# ANALYTICAL INTENTS
# =========================

        if (
            "compare" in q
            and len(float_ids) >= 2
        ):

            return {
                "type": "float_comparison",
                "float_ids": float_ids
            }

        if (
            "trend" in q
            or "warming" in q
            or "cooling" in q
        ):

            return {
                "type": "trend_analysis",
                "float_id": float_id
            }

        if any(
            keyword in q
            for keyword in [
                "anomaly",
                "anomalies",
                "outlier",
                "outliers",
                "unusual",
                "abnormal",
                "unexpected"
            ]
        ):

            return {
                "type": "anomaly_detection",
                "float_id": float_id
            }

        if any(
            keyword in q
            for keyword in [
                "water column",
                "thermocline",
                "stratification",
                "ocean layers",
                "vertical structure",
                "water profile",
                "column structure"
            ]
        ):

            return {
                "type": "water_column",
                "float_id": float_id
            }

        if any(
            keyword in q
            for keyword in [
                "mixed layer",
                "mixed layer depth",
                "mld",
                "calculate mld",
                "estimate mld",
                "determine mld",
                "mixed layer thickness",
                "surface layer depth"
            ]
        ):

            return {
                "type": "mixed_layer_depth",
                "float_id": float_id
            }

        if any(
            keyword in q
            for keyword in [
                "salinity analysis",
                "analyze salinity",
                "salinity structure",
                "halocline"
            ]
        ):

            return {
                "type": "salinity_analysis",
                "float_id": float_id
            }

        if any(
            keyword in q
            for keyword in [
                "oxygen analysis",
                "analyze oxygen",
                "oxygen structure"
            ]
        ):

            return {
                "type": "oxygen_analysis",
                "float_id": float_id
            }

        if any(
            keyword in q
            for keyword in [
                "oxygen minimum zone",
                "omz",
                "detect omz"
            ]
        ):

            return {
                "type": "omz_detection",
                "float_id": float_id
            }

        if any(
            keyword in q
            for keyword in [
                "trajectory",
                "float path",
                "drift",
                "float movement",
                "track float"
            ]
        ):

            return {
                "type": "trajectory_analysis",
                "float_id": float_id
            }

        # =========================
        # RETRIEVAL INTENTS
        # =========================

        if "latest profile" in q:

            return {
                "type": "latest_profile",
                "float_id": float_id
            }

        if any(
            keyword in q
            for keyword in [
                "temperature",
                "temperature profile",
                "temp profile"
            ]
        ):

            return {
                "type": "temperature_profile",
                "float_id": float_id
            }

        if "salinity" in q:

            return {
                "type": "salinity_profile",
                "float_id": float_id
            }

        if "oxygen" in q:

            return {
                "type": "oxygen_profile",
                "float_id": float_id
            }

        if "summary" in q:

            return {
                "type": "float_summary",
                "float_id": float_id
            }

        return {

            "type":
                "unknown",

            "float_id":
                float_id
        }