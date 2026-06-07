from app.services.llm.gemini_provider import (
    GeminiProvider
)


class SummaryService:

    def __init__(self):

        self.llm = GeminiProvider()

    async def summarize_temperature_profile(
        self,
        float_id: str,
        stats: dict
    ):

        prompt = f"""
You are an oceanographic analyst.

Generate a short scientific summary.

Float ID: {float_id}

Statistics:

{stats}

Write 2-3 concise sentences.
"""

        return await self.llm.generate(prompt)
    
    async def summarize_temperature_trend(
            self,
            float_id: str,
            trend: dict
        ):

            prompt = f"""
        You are an oceanographic analyst.

        Generate a concise scientific interpretation
        of the temperature trend.

        Float ID: {float_id}

        Trend Analysis:

        {trend}

        Explain:

        - Is the float warming or cooling?
        - How significant is the change?
        - What does this imply?

        Write 2-3 concise sentences.
        """

            return await self.llm.generate(
                prompt
            )
    async def summarize_float_comparison(
        self,
        float_a: str,
        float_b: str,
        comparison: dict
    ):

        prompt = f"""
    You are an oceanographic analyst.

    Compare the following ARGO floats.

    Float A: {float_a}

    Float B: {float_b}

    Comparison Data:

    {comparison}

    Write a concise scientific comparison.
    """

        return await self.llm.generate(
            prompt
        )
    async def summarize_anomalies(
        self,
        float_id: str,
        anomalies: dict
    ):

        prompt = f"""
        You are an expert oceanographer specializing in:

        - Argo floats
        - Ocean temperature
        - Salinity
        - Dissolved oxygen
        - Mixed layer depth
        - Ocean circulation

        Float ID:

        {float_id}

        Anomaly Results:

        {anomalies}

        IMPORTANT:

        - Use ONLY information present in the anomaly results.
        - Do NOT invent values.
        - Do NOT discuss unrelated scientific fields.
        - Focus only on oceanographic observations.

        Generate a concise oceanographic summary
        in 3-5 sentences.

        Output plain text only.
        """

        return await self.llm.generate(
            prompt
        )
    async def summarize_water_column(
        self,
        float_id: str,
        structure: dict
    ):

        prompt = f"""
    You are an oceanographic analyst.

    Interpret the vertical water-column
    structure.

    Float ID: {float_id}

    Structure Data:

    {structure}

    Discuss:

    - Surface conditions
    - Deep water conditions
    - Thermocline depth

    Write 2-3 concise sentences.
    """

        return await self.llm.generate(
            prompt
        )
    async def summarize_generated_results(
            self,
            query: str,
            rows
        ):

            prompt = f"""
        You are an oceanographic analyst.

        User Query:

        {query}

        Returned Data:

        {rows}

        Write a concise summary
        of the results.
        """

            return await self.llm.generate(
                prompt
            )
    async def summarize_analysis(
            self,
            analysis_type,
            analysis
        ):

            prompt = f"""
You are an oceanographic analyst.

Analysis Type:
{analysis_type}

Results:
{analysis}

Rules:

- Use only metrics present in Results.
- Explain what the metrics indicate.
- Do not mention variables not present.
- Do not invent causes.
- Do not infer biological, chemical, or physical processes unless explicitly supported by the metrics.

Generate a concise oceanographic summary.
"""

            return await (
                self.llm.generate(
                    prompt
                )
            )
    async def summarize_generated_results(
            self,
            query: str,
            rows
        ):

            prompt = f"""
        You are an oceanographic analyst.

        User Query:

        {query}

        Returned Data:

        {rows}

        Instructions:

        - Answer naturally.
        - Focus on useful findings.
        - Do not discuss databases, SQL, retrieval, datasets, missing information, or system limitations.
        - Do not say "no results", "insufficient data", "cannot determine", "data not available", or similar phrases.
        - Provide a concise scientific response.

        Output plain text only.
        """

            return await self.llm.generate(
                prompt
            )