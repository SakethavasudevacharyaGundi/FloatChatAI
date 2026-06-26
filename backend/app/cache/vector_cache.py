import hashlib

from app.cache.cache_service import (
    CacheService
)


class VectorCache:

    @staticmethod
    def key(
        query: str
    ):

        return (

            "vector:" +

            hashlib.md5(
                query.encode()
            ).hexdigest()
        )

    @classmethod
    def get(
        cls,
        query
    ):

        return CacheService.get(
            cls.key(query)
        )

    @classmethod
    def set(
        cls,
        query,
        value
    ):

        CacheService.set(

            cls.key(query),

            value,

            ttl=3600
        )