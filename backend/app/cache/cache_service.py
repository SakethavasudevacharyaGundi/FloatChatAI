import json
from datetime import datetime, date

from app.cache.redis_client import (
    redis_client
)


def json_serializer(obj):

    if isinstance(
        obj,
        (datetime, date)
    ):
        return obj.isoformat()

    return str(obj)


class CacheService:

    @staticmethod
    def get(key):

        value = redis_client.get(key)

        if not value:
            return None

        return json.loads(value)

    @staticmethod
    def set(
        key,
        value,
        ttl=1800
    ):

        redis_client.setex(

            key,

            ttl,

            json.dumps(
                value,
                default=json_serializer
            )
        )