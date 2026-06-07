import re


class InputValidator:

    MAX_QUERY_LENGTH = 1000

    MIN_QUERY_LENGTH = 0

    MAX_FLOAT_IDS = 10

    FORBIDDEN_PATTERNS = [

        r";",

        r"--",

        r"/\*",

        r"\*/",

        r"\bDROP\b",

        r"\bDELETE\b",

        r"\bUPDATE\b",

        r"\bINSERT\b",

        r"\bALTER\b",

        r"\bTRUNCATE\b"
    ]

    @classmethod
    def validate(
        cls,
        query: str
    ):

        if not query:

            raise Exception(
                "Query cannot be empty."
            )

        query = query.strip()

        if len(query) < cls.MIN_QUERY_LENGTH:

            raise Exception(
                "Query too short."
            )

        if len(query) > cls.MAX_QUERY_LENGTH:

            raise Exception(
                "Query too long."
            )

        for pattern in cls.FORBIDDEN_PATTERNS:

            if re.search(
                pattern,
                query,
                re.IGNORECASE
            ):

                raise Exception(
                    "Unsafe query detected."
                )

        return True