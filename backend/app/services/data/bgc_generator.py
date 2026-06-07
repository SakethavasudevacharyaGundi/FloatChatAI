import numpy as np
from typing import Dict, Any


class BGCGenerator:

    """
    Generate synthetic Bio-Geochemical parameters
    from physical oceanographic variables.
    """

    def generate(
        self,
        temperature: float,
        salinity: float,
        pressure: float
    ) -> Dict[str, Any]:

        oxygen = max(
            50,
            300 - (temperature * 5) - (pressure * 0.02)
        )

        chlorophyll = max(
            0.01,
            2.5 - (pressure * 0.002)
        )

        nitrate = max(
            0.1,
            (pressure * 0.01) + (35 - salinity)
        )

        ph = max(
            7.5,
            8.2 - (pressure * 0.0001)
        )

        return {
            "oxygen_umol_kg": round(oxygen, 2),
            "chlorophyll_mg_m3": round(chlorophyll, 3),
            "nitrate_umol_kg": round(nitrate, 3),
            "ph_total": round(ph, 3)
        }


bgc_generator = BGCGenerator()