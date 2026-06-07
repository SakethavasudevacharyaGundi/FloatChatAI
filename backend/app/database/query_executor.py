from sqlalchemy import text

from app.database.connection import engine


class QueryExecutor:

    def execute(self, sql: str):

        with engine.connect() as conn:

            result = conn.execute(
                text(sql)
            )

            return [
                dict(row._mapping)
                for row in result
            ]