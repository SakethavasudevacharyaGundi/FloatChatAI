class StatisticsService:

    def summarize_temperature_profile(self, rows):

        if not rows:
            return {}

        temperatures = [
            row["temperature_c"]
            for row in rows
            if row.get("temperature_c") is not None
        ]

        depths = [
            row["depth_m"]
            for row in rows
            if row.get("depth_m") is not None
        ]

        return {
            "count": len(rows),

            "min_temp": min(temperatures),
            "max_temp": max(temperatures),
            "avg_temp": round(
                sum(temperatures) / len(temperatures),
                2
            ),

            "min_depth": min(depths),
            "max_depth": max(depths)
        }