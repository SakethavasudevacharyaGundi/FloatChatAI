from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from app.api.routes.query import router as query_router
from app.services.llm.gemini_provider import (
    GeminiProvider
)
from app.database.query_executor import QueryExecutor
from app.cache.redis_client import (
    redis_client
)
limiter = Limiter(
    key_func=get_remote_address
)
app = FastAPI(
    title="FloatChatAI",
    version="1.0.0"
)
app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

app.add_middleware(
    SlowAPIMiddleware
)
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(query_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "FloatChatAI Backend Running"
    }

# Health endpoint

@app.get("/health")
async def health():

    db_status = "ok"

    redis_status = "ok"

    llm_status = "ok"

    try:

        executor = QueryExecutor()

        executor.execute(
            "SELECT 1"
        )

    except Exception:

        db_status = "failed"

    try:

        redis_client.ping()

    except Exception:

        redis_status = "failed"

    try:

        import requests

        llm_status = "ok"

        try:

            requests.get(
                "http://localhost:11434/api/tags",
                timeout=3
            )

        except Exception:

            llm_status = "failed"

    except Exception:

        llm_status = "failed"

    overall = (

        "healthy"

        if all(

            status == "ok"

            for status in [

                db_status,
                redis_status,
                llm_status
            ]
        )

        else

        "degraded"
    )

    return {

        "status": overall,

        "database": db_status,

        "redis": redis_status,

        "llm": llm_status
    }
@app.get("/test-llm")
async def test_llm():

    provider = GeminiProvider()

    result = await provider.generate(
        "What is ocean salinity?"
    )

    return {
        "response": result
    }
@app.get("/test-db")
async def test_db():

    executor = QueryExecutor()

    rows = executor.execute(
        "SELECT 1 as test"
    )

    return rows

# System status
@app.get("/ready")
async def ready():

    try:

        QueryExecutor().execute(
            "SELECT 1"
        )

        redis_client.ping()

        return {
            "ready": True
        }

    except Exception:

        return {
            "ready": False
        }
@app.get("/live")
async def live():

    return {
        "alive": True
    }