import numpy as np

class ComparisonService:

    def compare_temperature(
        self,
        float_a_rows,
        float_b_rows
    ):

        temps_a = [
            r["temperature_c"]
            for r in float_a_rows
            if r["temperature_c"] is not None
        ]

        temps_b = [
            r["temperature_c"]
            for r in float_b_rows
            if r["temperature_c"] is not None
        ]

        avg_a = np.mean(temps_a)
        avg_b = np.mean(temps_b)

        return {

            "avg_temp_a":
                round(float(avg_a), 2),

            "avg_temp_b":
                round(float(avg_b), 2),

            "min_temp_a":
                round(float(min(temps_a)), 2),

            "min_temp_b":
                round(float(min(temps_b)), 2),

            "max_temp_a":
                round(float(max(temps_a)), 2),

            "max_temp_b":
                round(float(max(temps_b)), 2),

            "difference":
                round(float(avg_a - avg_b), 2)
        }

    def compare_profile_counts(
        self,
        profiles_a,
        profiles_b
    ):

        return {

            "profiles_a":
                len(profiles_a),

            "profiles_b":
                len(profiles_b),

            "difference":
                len(profiles_a)
                - len(profiles_b)
        }

    def compare_depths(
        self,
        float_a_rows,
        float_b_rows
    ):

        depths_a = [
            r["depth_m"]
            for r in float_a_rows
            if r["depth_m"] is not None
        ]

        depths_b = [
            r["depth_m"]
            for r in float_b_rows
            if r["depth_m"] is not None
        ]

        return {

            "max_depth_a":
                round(float(max(depths_a)), 2),

            "max_depth_b":
                round(float(max(depths_b)), 2),

            "difference":
                round(
                    float(
                        max(depths_a)
                        -
                        max(depths_b)
                    ),
                    2
                )
        }

    def compare(
        self,
        float_a_rows,
        float_b_rows,
        profiles_a,
        profiles_b
    ):

        return {

            "temperature":
                self.compare_temperature(
                    float_a_rows,
                    float_b_rows
                ),

            "profiles":
                self.compare_profile_counts(
                    profiles_a,
                    profiles_b
                ),  

            "depth":
                self.compare_depths(
                    float_a_rows,
                    float_b_rows
                )
        }
    def compare_from_rows(
        self,
        rows
        ):


        if len(rows) != 2:

            return {
                "error":
                    "Need exactly two floats."
            }

        a = rows[0]
        b = rows[1]

        return {

            "float_a":
                a["float_id"],

            "float_b":
                b["float_id"],

            "avg_temperature_a":
                round(
                    float(
                        a["avg_temperature"]
                    ),
                    2
                ),

            "avg_temperature_b":
                round(
                    float(
                        b["avg_temperature"]
                    ),
                    2
                ),

            "avg_salinity_a":
                round(
                    float(
                        a["avg_salinity"]
                    ),
                    2
                ),

            "avg_salinity_b":
                round(
                    float(
                        b["avg_salinity"]
                    ),
                    2
                ),

            "max_depth_a":
                round(
                    float(
                        a["max_depth"]
                    ),
                    2
                ),

            "max_depth_b":
                round(
                    float(
                        b["max_depth"]
                    ),
                    2
                ),

            "profile_count_a":
                a["profile_count"],

            "profile_count_b":
                b["profile_count"]
        }