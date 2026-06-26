from app.database.query_executor import QueryExecutor


class ProfileRepository:

    def __init__(self):

        self.executor = QueryExecutor()

    def get_latest_profile(
        self,
        float_id: str
    ):

        sql = f"""
        SELECT *
        FROM profiles
        WHERE float_id = '{float_id}'
        ORDER BY profile_datetime DESC
        LIMIT 1
        """

        rows = self.executor.execute(sql)

        return rows[0] if rows else None