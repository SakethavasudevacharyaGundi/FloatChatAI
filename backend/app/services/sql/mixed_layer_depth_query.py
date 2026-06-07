class MixedLayerDepthQuery:

    @staticmethod
    def build(
        profile_id
    ):

        return """

        SELECT

            depth_m,
            temperature_c

        FROM measurements

        WHERE profile_id = ?

        ORDER BY depth_m ASC

        """