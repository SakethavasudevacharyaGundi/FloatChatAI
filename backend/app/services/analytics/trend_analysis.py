import numpy as np
from scipy.stats import linregress


class TrendAnalysisService:

    def analyze_temperature_trend(self, rows):

        if len(rows) < 2:
            return {
                "direction": "insufficient_data"
            }

        temps = [
            row["temperature_c"]
            for row in rows
            if row["temperature_c"] is not None
        ]

        if len(temps) < 2:
            return {
                "direction": "insufficient_data"
            }

        x = np.arange(len(temps))

        slope, intercept, r_value, p_value, std_err = (
            linregress(x, temps)
        )

        direction = (
            "warming"
            if slope > 0
            else "cooling"
        )
        dates = []

        for i in range(len(temps)):

            dates.append(i)
        return {
            "direction": direction,
            "slope": round(float(slope), 4),
            "r_squared": round(
                float(r_value ** 2),
                4
            ),
            "change": round(
                float(temps[-1] - temps[0]),
                4
            ),
            "start_temp": temps[0],
            "end_temp": temps[-1],
            "dates": dates,
            "temperatures": temps
        }