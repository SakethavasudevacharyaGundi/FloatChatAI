import numpy as np


class AnomalyDetector:

    def detect(
        self,
        rows
    ):

        return self.detect_temperature_anomalies(
            rows
        )

    def detect_temperature_anomalies(
        self,
        rows
    ):

        temps = np.array([
            row["temperature_c"]
            for row in rows
            if row["temperature_c"] is not None
        ])

        if len(temps) < 1:

            return {
                "anomaly_count": 0,
                "anomalies": []
            }

        mean = np.mean(temps)
        std = np.std(temps)

        anomalies = []

        for row in rows:

            temp = row.get(
                "temperature_c"
            )

            if temp is None:
                continue

            z_score = (
                temp - mean
            ) / std

            if abs(z_score) > 3:

                anomalies.append({

                    "temperature_c":
                        temp,

                    "z_score":
                        round(
                            float(z_score),
                            2
                        )
                })

        return {

            "anomaly_count":
                len(anomalies),

            "anomalies":
                anomalies[:20]
        }