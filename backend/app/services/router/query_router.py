class QueryRouter:

    def route(self, query: str) -> str:

        q = query.lower()

        # Vector Search Queries
        if "similar" in q:
            return "vector"

        # SQL Queries
        if "temperature" in q:
            return "sql"

        if "salinity" in q:
            return "sql"

        if "oxygen" in q:
            return "sql"

        if "chlorophyll" in q:
            return "sql"

        if "float" in q:
            return "sql"

        # Everything else
        return "general"