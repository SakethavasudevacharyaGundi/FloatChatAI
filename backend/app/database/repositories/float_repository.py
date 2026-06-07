from app.database.query_executor import QueryExecutor


class FloatRepository:

    def __init__(self):

        self.executor = QueryExecutor()

    def get_float(
        self,
        float_id: str
    ):

        sql = f"""
        SELECT *
        FROM floats
        WHERE float_id = '{float_id}'
        """

        rows = self.executor.execute(sql)

        return rows[0] if rows else None
    def get_float_summary(
    self,
    float_id: str
        ):

            sql = f"""
            SELECT
                f.float_id,
                f.region,
                f.status,

                COUNT(p.profile_id) as profile_count,

                MIN(p.profile_datetime) as first_profile,

                MAX(p.profile_datetime) as latest_profile

            FROM floats f

            LEFT JOIN profiles p
                ON f.float_id = p.float_id

            WHERE f.float_id = '{float_id}'

            GROUP BY
                f.float_id,
                f.region,
                f.status
            """

            rows = self.executor.execute(sql)

            return rows[0] if rows else None