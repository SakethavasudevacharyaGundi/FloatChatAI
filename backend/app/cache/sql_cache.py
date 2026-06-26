import hashlib

from app.cache.cache_service import (
    CacheService
)


class SQLCache:

    @staticmethod
    def build_key(
        sql: str
    ):

        return (

            "sql:" +

            hashlib.md5(
                sql.encode()
            ).hexdigest()
        )

    @classmethod
    def get(
        cls,
        sql: str
    ):

        return CacheService.get(
            cls.build_key(sql)
        )

    @classmethod
    def set(
        cls,
        sql: str,
        rows
    ):

        CacheService.set(

            cls.build_key(sql),

            rows,

            ttl=1800
        )