from fastapi import APIRouter

from app.services.pipeline.query_pipeline import QueryPipeline

from app.schemas.data_models import QueryRequest

router = APIRouter()

pipeline = None

@router.post("/demo-query")
async def query_endpoint(payload: QueryRequest):

    global pipeline

    if pipeline is None:
        pipeline = QueryPipeline()

    result = await pipeline.process(payload.query)

    return result