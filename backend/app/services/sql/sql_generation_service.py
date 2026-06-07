from app.services.llm.gemini_provider import (
    GeminiProvider
)

import re


class SQLGenerationService:

    def __init__(self):

        self.llm = GeminiProvider()

        self.schema = """
Table: measurements

Columns:
- profile_id
- depth_m
- temperature_c
- salinity_psu
- oxygen_umol_kg

Table: profiles

Columns:
- profile_id
- float_id
- profile_datetime
- lat
- lon
- cycle_number

Table: floats

Columns:
- float_id
- region
- status
- total_profiles
"""

    async def generate_sql(
        self,
        query: str,
        intent_type: str = None
    ):

        # =========================
        # ANALYTICAL PROMPTS
        # =========================

        if intent_type == "trend_analysis":

            prompt = f"""
You are a PostgreSQL expert.

Database Schema:

{self.schema}

Generate SQL for trend analysis.

Return ONLY these columns:

- profile_datetime
- temperature_c

User Query:

{query}

Output ONLY SQL.
No markdown.
No explanations.
"""

        elif intent_type == "anomaly_detection":

            prompt = f"""
You are a PostgreSQL expert.

Database Schema:

{self.schema}

Generate SQL for anomaly detection.

Return ONLY:

- profile_datetime
- depth_m
- temperature_c

User Query:

{query}
IMPORTANT:

- float_id is VARCHAR
- Always quote float IDs

Correct:
WHERE p.float_id = '1900975'

Incorrect:
WHERE p.float_id = 1900975

Output ONLY SQL.
No markdown.
No explanations.
"""
        elif intent_type == "water_column":

            prompt = f"""
        You are a PostgreSQL expert.

        Database Schema:

        {self.schema}

        Generate SQL for water column analysis.

        Requirements:

        - Return ONLY:
            depth_m
            temperature_c
            salinity_psu

        - Use:
            measurements m
            profiles p

        - Join:
            m.profile_id = p.profile_id

        - float_id is VARCHAR

        IMPORTANT:

        Correct:
        WHERE p.float_id = '1900975'

        Incorrect:
        WHERE p.float_id = 1900975

        - Always quote float_id values.
        - Use p.float_id in the WHERE clause.
        - Order results by depth_m ascending.
        - Do not use aggregation.
        - Do not use GROUP BY.
        - Do not use subqueries.
        - Do not return any columns other than:
            depth_m
            temperature_c
            salinity_psu

        Example SQL:

        SELECT
            m.depth_m,
            m.temperature_c,
            m.salinity_psu
        FROM measurements m
        JOIN profiles p
            ON m.profile_id = p.profile_id
        WHERE p.float_id = '1900975'
        ORDER BY m.depth_m;

        User Query:

        {query}

        Output ONLY valid PostgreSQL SQL.
        No markdown.
        No explanations.
        No code fences.
        """

        elif intent_type == "float_comparison":

            prompt = f"""
        You are a PostgreSQL expert.

        Database Schema:

        {self.schema}

        Generate SQL for comparing floats.

        Requirements:

        - Use profiles alias p
        - Use measurements alias m
        - Join on m.profile_id = p.profile_id

        Return exactly:

        p.float_id,
        AVG(m.temperature_c) AS avg_temperature,
        AVG(m.salinity_psu) AS avg_salinity,
        MAX(m.depth_m) AS max_depth,
        COUNT(DISTINCT p.profile_id) AS profile_count

        IMPORTANT:
        - float_id is VARCHAR
        - Always quote float ids
        - Use p.float_id in WHERE
        - Use p.profile_id in COUNT
        - Group by p.float_id

        User Query:
        {query}

        Output SQL only.
        """
        elif intent_type == "mixed_layer_depth":

            prompt = f"""
        You are a PostgreSQL expert.

        Database Schema:

        {self.schema}

        Generate SQL for mixed layer depth analysis.

        Return ONLY:

        - depth_m
        - temperature_c

        Use:
        - measurements m
        - profiles p

        Join:
        m.profile_id = p.profile_id

        IMPORTANT:

        - float_id is VARCHAR

        Correct:

        WHERE p.float_id = '1900975'

        Incorrect:

        WHERE p.float_id = 1900975

        Order by:
        m.depth_m ASC

        Output ONLY valid PostgreSQL SQL.

        User Query:

        {query}
        """

        # =========================
        # GENERIC FALLBACK
        # =========================
        elif intent_type == "salinity_analysis":

             prompt = f"""
        You are a PostgreSQL expert.

        Database Schema:

        {self.schema}

        Generate SQL for salinity analysis.

        Return ONLY:

        - depth_m
        - salinity_psu

        Use:
        - measurements m
        - profiles p

        Join:
        m.profile_id = p.profile_id

        IMPORTANT:

        - float_id is VARCHAR

        Correct:

        WHERE p.float_id = '1900975'

        Incorrect:

        WHERE p.float_id = 1900975

        Order by:
        m.depth_m ASC

        Output ONLY valid PostgreSQL SQL.

        User Query:

        {query}
        """
        elif intent_type == "oxygen_analysis":

            prompt = f"""
        You are a PostgreSQL expert.

        Database Schema:

        {self.schema}

        Generate SQL for oxygen analysis.

        Return ONLY:

        - depth_m
        - oxygen_umol_kg

        Use:
        - measurements m
        - profiles p

        Join:
        m.profile_id = p.profile_id

        IMPORTANT:

        - float_id is VARCHAR

        Correct:

        WHERE p.float_id = '1900975'

        Incorrect:

        WHERE p.float_id = 1900975

        Order by:
        m.depth_m ASC

        Output ONLY valid PostgreSQL SQL.

        User Query:

        {query}
        """
        elif intent_type == "omz_detection":

            prompt = f"""
        You are a PostgreSQL expert.

        Database Schema:

        {self.schema}

        Generate SQL for oxygen minimum zone detection.

        Return ONLY:

        - depth_m
        - oxygen_umol_kg

        Use:
        - measurements m
        - profiles p

        Join:
        m.profile_id = p.profile_id

        IMPORTANT:

        - float_id is VARCHAR

        Correct:

        WHERE p.float_id = '1900975'

        Incorrect:

        WHERE p.float_id = 1900975

        Order by:
        oxygen_umol_kg ASC

        Output ONLY valid PostgreSQL SQL.

        User Query:

        {query}
        """
        elif intent_type == "trajectory_analysis":

            prompt = f"""
        You are a PostgreSQL expert.

        Database Schema:

        {self.schema}

        Generate SQL for float trajectory analysis.

        Return ONLY:

        - profile_datetime
        - lat
        - lon
        - cycle_number

        Use ONLY:
        profiles

        IMPORTANT:

        - float_id is VARCHAR

        Correct:

        WHERE float_id = '1900975'

        Incorrect:

        WHERE float_id = 1900975

        Order by:
        profile_datetime ASC

        Output ONLY valid PostgreSQL SQL.

        User Query:

        {query}
        ONLY valid PostgreSQL SQL.
        """
        else:

            prompt = f"""
You are a PostgreSQL expert.

Database Schema:

{self.schema}

IMPORTANT:

- float_id is VARCHAR
- profile_id is VARCHAR
- Always quote string values

Example:

WHERE p.float_id = '1900975'

IMPORTANT:

Only use columns listed in schema.

Do not invent columns.

Generate ONLY valid PostgreSQL SQL.

Output ONLY SQL.
No markdown.
No explanations.

User Query:

{query}
"""

        sql = await self.llm.generate(
            prompt
        )
        float_ids = re.findall(
        r"\b\d{7}\b",
        query
        )

        if float_ids:

            float_id = float_ids[0]

            sql = re.sub(
                rf"p\.float_id\s*=\s*{float_id}",
                f"p.float_id = '{float_id}'",
                sql
            )

            sql = re.sub(
                rf"float_id\s*=\s*{float_id}",
                f"float_id = '{float_id}'",
                sql
            )

        # Remove markdown fences

        sql = re.sub(
            r"```sql|```",
            "",
            sql,
            flags=re.IGNORECASE
        ).strip()

        # Extract first SQL statement

        match = re.search(
            r"(SELECT[\s\S]*?;)",
            sql,
            re.IGNORECASE
        )

        if match:

            sql = match.group(1)

        return sql.strip()