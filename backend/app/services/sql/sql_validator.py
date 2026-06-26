import re
from app.services.security.sql_whitelist import (
    SQLWhitelist
)

class SQLValidator:
    MAX_LIMIT = 1000

    FORBIDDEN_KEYWORDS = [

        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE"

    ]

    ALLOWED_TABLES = [

        "floats",
        "profiles",
        "measurements"

    ]

    @classmethod
    def validate(
        cls,
        sql: str
    ):

        sql_upper = sql.upper()

        # ==========================
        # BLOCK DANGEROUS COMMANDS
        # ==========================

        for keyword in cls.FORBIDDEN_KEYWORDS:

            if re.search(
                rf"\b{keyword}\b",
                sql_upper
            ):

                raise Exception(
                    f"Forbidden SQL command detected: {keyword}"
                )

        # ==========================
        # ONLY ALLOW SELECT / WITH
        # ==========================

        sql_start = sql.strip().upper()

        if not (
            sql_start.startswith("SELECT")
            or sql_start.startswith("WITH")
        ):

            raise Exception(
                "Only SELECT and WITH queries are allowed."
            )

        # ==========================
        # VALIDATE TABLE NAMES
        # ==========================

        tables = re.findall(
            r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            sql,
            re.IGNORECASE
        )

        for table in tables:

            if table.lower() not in SQLWhitelist.ALLOWED_TABLES:

                raise Exception(
                    f"Unknown table referenced: {table}"
                )
        limit_match = re.search(
            r"LIMIT\s+(\d+)",
            sql,
            re.IGNORECASE
        )

        if limit_match:

            limit_value = int(
                limit_match.group(1)
            )

            if limit_value > cls.MAX_LIMIT:

                raise Exception(
                    f"LIMIT exceeds maximum allowed rows ({cls.MAX_LIMIT})."
                )

        return True