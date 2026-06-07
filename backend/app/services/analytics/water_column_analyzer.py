import numpy as np


class WaterColumnAnalyzer:

    def analyze(
        self,
        rows
    ):

        if len(rows) < 1:

            return {
                "status":
                    "insufficient_data"
            }

        sorted_rows = sorted(
            rows,
            key=lambda r: r["depth_m"]
        )

        surface = sorted_rows[:10]

        deep = sorted_rows[-10:]

        surface_temp = np.mean([
            r["temperature_c"]
            for r in surface
            if r["temperature_c"] is not None
        ])

        deep_temp = np.mean([
            r["temperature_c"]
            for r in deep
            if r["temperature_c"] is not None
        ])

        surface_salinity = np.mean([
            r["salinity_psu"]
            for r in surface
            if r.get("salinity_psu") is not None
        ])

        deep_salinity = np.mean([
            r["salinity_psu"]
            for r in deep
            if r.get("salinity_psu") is not None
        ])

        thermocline_depth = (
            self.find_thermocline(
                sorted_rows
            )
        )
        depths = []

        temperatures = []

        for row in sorted_rows:

            if row["temperature_c"] is None:
                continue

            depths.append(
                row["depth_m"]
            )

            temperatures.append(
                row["temperature_c"]
            )

        return {

            "surface_temp":
                round(
                    float(surface_temp),
                    2
                ),

            "deep_temp":
                round(
                    float(deep_temp),
                    2
                ),

            "temperature_change":
                round(
                    float(
                        deep_temp
                        - surface_temp
                    ),
                    2
                ),

            "surface_salinity":
                round(
                    float(surface_salinity),
                    2
                ),

            "deep_salinity":
                round(
                    float(deep_salinity),
                    2
                ),

            "thermocline_depth":
                thermocline_depth,
            
            "depths":
                depths,
            "temperatures":
                temperatures
        }

    def find_thermocline(
        self,
        rows
    ):

        largest_gradient = 0

        thermocline_depth = None

        for i in range(
            len(rows) - 1
        ):

            t1 = rows[i][
                "temperature_c"
            ]

            t2 = rows[i + 1][
                "temperature_c"
            ]

            d1 = rows[i][
                "depth_m"
            ]

            d2 = rows[i + 1][
                "depth_m"
            ]

            if (
                t1 is None
                or t2 is None
            ):
                continue

            delta_depth = d2 - d1

            if delta_depth == 0:
                continue

            gradient = abs(
                (t2 - t1)
                / delta_depth
            )

            if (
                gradient
                > largest_gradient
            ):

                largest_gradient = (
                    gradient
                )

                thermocline_depth = (
                    d1
                )
            
        return (
            round(
                float(thermocline_depth),
                2
            )
            if thermocline_depth
            is not None
            else None
        )