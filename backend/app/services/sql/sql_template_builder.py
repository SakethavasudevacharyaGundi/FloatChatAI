import re


class SQLTemplateBuilder:

    def build(self, query: str, intent: dict):

        match = re.search(
            r'(\d{7})',
            query
        )

        float_id = match.group(1) if match else None

        if not float_id:
            return None

        if intent["type"] == "temperature_profile":

            return f"""
            SELECT
                m.depth_m,
                m.temperature_c
            FROM measurements m
            JOIN profiles p
                ON m.profile_id = p.profile_id
            WHERE p.float_id = '{float_id}'
            ORDER BY m.depth_m
            """

        return None