"""
Enhanced FastAPI Backend with Full MCP Integration
Provides comprehensive API endpoints for ARGO oceanographic data analysis
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our enhanced components
from models.data_models import QueryRequest, QueryResponse
from rag.mcp_enhanced_rag import enhanced_rag_pipeline, process_query_enhanced
from mcp.mcp_client import mcp_manager, get_mcp_context_for_gemini, process_query_with_mcp
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="🌊 ARGO Ocean Data API - MCP Enhanced",
    description="Advanced oceanographic data analysis API with Model Context Protocol integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
system_initialized = False
system_stats = {
    "start_time": datetime.now(),
    "total_queries": 0,
    "mcp_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0
}

# Pydantic models for API
class SystemStatus(BaseModel):
    """System status response model"""
    status: str
    mcp_enabled: bool
    rag_pipeline_initialized: bool
    gemini_model: str
    uptime_seconds: float
    total_queries: int
    mcp_queries: int
    success_rate: float

class MCPToolRequest(BaseModel):
    """MCP tool request model"""
    tool_name: str
    arguments: Dict[str, Any] = {}

class MCPToolResponse(BaseModel):
    """MCP tool response model"""
    success: bool
    tool_name: str
    data: Any = None
    error: str = None
    execution_time: float

class BulkQueryRequest(BaseModel):
    """Bulk query request model"""
    queries: List[str]
    parallel: bool = True

class BulkQueryResponse(BaseModel):
    """Bulk query response model"""
    total_queries: int
    successful: int
    failed: int
    results: List[QueryResponse]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global system_initialized
    
    logger.info("🚀 Starting Enhanced ARGO API with MCP integration...")
    
    try:
        # Initialize MCP manager
        mcp_success = await mcp_manager.initialize()
        logger.info(f"MCP Manager initialized: {mcp_success}")
        
        # Initialize enhanced RAG pipeline
        rag_success = await enhanced_rag_pipeline.initialize()
        logger.info(f"Enhanced RAG Pipeline initialized: {rag_success}")
        
        system_initialized = True
        logger.info("✅ System initialization completed successfully")
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        system_initialized = False

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup system components on shutdown"""
    logger.info("🔄 Shutting down Enhanced ARGO API...")
    
    try:
        await enhanced_rag_pipeline.cleanup()
        await mcp_manager.cleanup()
        logger.info("✅ System cleanup completed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Health check endpoint
@app.get("/health", response_model=SystemStatus)
async def health_check():
    """System health and status endpoint"""
    uptime = (datetime.now() - system_stats["start_time"]).total_seconds()
    success_rate = (
        system_stats["successful_queries"] / max(system_stats["total_queries"], 1) * 100
    )
    
    return SystemStatus(
        status="healthy" if system_initialized else "initializing",
        mcp_enabled=mcp_manager.client.initialized if hasattr(mcp_manager, 'client') else False,
        rag_pipeline_initialized=enhanced_rag_pipeline.mcp_enabled,
        gemini_model=config.GEMINI_MODEL,
        uptime_seconds=uptime,
        total_queries=system_stats["total_queries"],
        mcp_queries=system_stats["mcp_queries"],
        success_rate=success_rate
    )

# Main query endpoint with enhanced processing
@app.post("/query/enhanced", response_model=QueryResponse)
async def process_enhanced_query(request: QueryRequest):
    """Process query using enhanced MCP + RAG pipeline"""
    system_stats["total_queries"] += 1
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing enhanced query: {request.query}")
        
        if not system_initialized:
            raise HTTPException(status_code=503, detail="System not fully initialized")
        
        # Process using enhanced pipeline
        response = await process_query_enhanced(request)
        
        # Update stats
        if response.metadata and response.metadata.get("mcp_tools_used"):
            system_stats["mcp_queries"] += 1
        
        system_stats["successful_queries"] += 1
        
        # Add execution metadata
        execution_time = (datetime.now() - start_time).total_seconds()
        response.metadata = response.metadata or {}
        response.metadata["api_execution_time"] = execution_time
        response.metadata["processed_at"] = datetime.now().isoformat()
        
        logger.info(f"Query processed successfully in {execution_time:.2f}s")
        return response
        
    except Exception as e:
        system_stats["failed_queries"] += 1
        logger.error(f"Error processing enhanced query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# MCP tool direct access endpoint
@app.post("/mcp/tool", response_model=MCPToolResponse)
async def execute_mcp_tool(request: MCPToolRequest):
    """Execute MCP tool directly"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Executing MCP tool: {request.tool_name}")
        
        if not mcp_manager.client.initialized:
            raise HTTPException(status_code=503, detail="MCP system not initialized")
        
        # Execute tool
        result = await mcp_manager.client.call_tool(request.tool_name, request.arguments)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return MCPToolResponse(
            success=result.success,
            tool_name=request.tool_name,
            data=result.data,
            error=result.error,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error executing MCP tool: {e}")
        execution_time = (datetime.now() - start_time).total_seconds()
        return MCPToolResponse(
            success=False,
            tool_name=request.tool_name,
            error=str(e),
            execution_time=execution_time
        )

# MCP tools list endpoint
@app.get("/mcp/tools")
async def list_mcp_tools():
    """Get list of available MCP tools"""
    try:
        if not mcp_manager.client.initialized:
            return {"error": "MCP system not initialized", "tools": []}
        
        tools_response = await mcp_manager.client.list_tools()
        
        if tools_response.success:
            return {
                "available": True,
                "tools": tools_response.data.get("tools", []),
                "count": len(tools_response.data.get("tools", []))
            }
        else:
            return {"error": tools_response.error, "tools": []}
            
    except Exception as e:
        logger.error(f"Error listing MCP tools: {e}")
        return {"error": str(e), "tools": []}

# MCP context endpoint for frontend integration
@app.get("/mcp/context")
async def get_mcp_context():
    """Get MCP context information for frontend"""
    try:
        if not mcp_manager.client.initialized:
            return {"available": False, "context": "MCP system not initialized"}
        
        context = await get_mcp_context_for_gemini()
        return {
            "available": True,
            "context": context,
            "tools_count": len(mcp_manager.client.available_tools)
        }
        
    except Exception as e:
        logger.error(f"Error getting MCP context: {e}")
        return {"available": False, "context": f"Error: {str(e)}"}

# Legacy endpoint for backward compatibility
@app.post("/query", response_model=QueryResponse)
async def process_query_legacy(request: QueryRequest):
    """Legacy query endpoint for backward compatibility"""
    return await process_enhanced_query(request)

# System statistics endpoint
@app.get("/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    uptime = (datetime.now() - system_stats["start_time"]).total_seconds()
    
    stats = {
        "system": {
            "uptime_seconds": uptime,
            "uptime_formatted": f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m {uptime%60:.0f}s",
            "initialized": system_initialized,
            "start_time": system_stats["start_time"].isoformat()
        },
        "queries": {
            "total": system_stats["total_queries"],
            "mcp_enabled": system_stats["mcp_queries"],
            "successful": system_stats["successful_queries"],
            "failed": system_stats["failed_queries"],
            "success_rate": system_stats["successful_queries"] / max(system_stats["total_queries"], 1) * 100
        },
        "components": {
            "mcp_manager_initialized": hasattr(mcp_manager, 'client') and mcp_manager.client.initialized,
            "rag_pipeline_initialized": enhanced_rag_pipeline.mcp_enabled,
            "gemini_model": config.GEMINI_MODEL,
            "api_version": "2.0.0"
        }
    }
    
    return stats

# Test MCP integration endpoint
@app.post("/test/mcp")
async def test_mcp_integration():
    """Test MCP integration with sample queries"""
    test_results = {}
    
    test_queries = [
        ("get_system_stats", {}),
        ("get_available_regions", {}),
        ("get_available_parameters", {}),
        ("query_argo_data", {"query": "temperature data", "limit": 5})
    ]
    
    for tool_name, args in test_queries:
        try:
            if mcp_manager.client.initialized:
                result = await mcp_manager.client.call_tool(tool_name, args)
                test_results[tool_name] = {
                    "success": result.success,
                    "error": result.error,
                    "has_data": result.data is not None
                }
            else:
                test_results[tool_name] = {"success": False, "error": "MCP not initialized"}
        except Exception as e:
            test_results[tool_name] = {"success": False, "error": str(e)}
    
    return {
        "mcp_initialized": mcp_manager.client.initialized if hasattr(mcp_manager, 'client') else False,
        "test_results": test_results
    }

if __name__ == "__main__":
    uvicorn.run(
        "api.mcp_enhanced_api:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )