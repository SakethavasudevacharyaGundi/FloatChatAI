from app.database.query_executor import QueryExecutor


class MeasurementRepository:

    def __init__(self):

        self.executor = QueryExecutor()

    def get_temperature_profile(
        self,
        float_id: str
    ):

        sql = f"""
        SELECT
            m.depth_m,
            m.temperature_c
        FROM measurements m
        JOIN profiles p
            ON m.profile_id = p.profile_id
        WHERE p.float_id = '{float_id}'
        ORDER BY m.depth_m
        """

        return self.executor.execute(sql)
    def get_temperature_trend(
            self,
            float_id: str
        ):

            sql = f"""
            SELECT
                p.profile_datetime,
                AVG(m.temperature_c) AS temperature_c
            FROM profiles p
            JOIN measurements m
                ON p.profile_id = m.profile_id
            WHERE p.float_id = '{float_id}'
            GROUP BY p.profile_datetime
            ORDER BY p.profile_datetime
            """

            return self.executor.execute(sql)