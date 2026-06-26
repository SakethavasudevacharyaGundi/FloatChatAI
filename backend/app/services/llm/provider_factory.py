# provider_factory.py

from app.services.llm.gemini_provider import (
    GeminiProvider
)


class ProviderFactory:

    @staticmethod
    def get_provider():

        return GeminiProvider()