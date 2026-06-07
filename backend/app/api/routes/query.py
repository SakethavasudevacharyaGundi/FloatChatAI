from fastapi import APIRouter

from app.services.pipeline.query_pipeline import QueryPipeline

from app.schemas.data_models import QueryRequest

router = APIRouter()

pipeline = QueryPipeline()


@router.post("/demo-query")
async def query_endpoint(payload: QueryRequest):

    result = await pipeline.process(
        payload.query
    )

    return result