#!/usr/bin/env python3
"""
Advanced Features API Extension
Extends the unified API with advanced export, analysis, and session management capabilities.
"""

from fastapi import APIRouter, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid
import logging
from io import BytesIO
import json

# Import our advanced modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export.advanced_exporter import advanced_exporter
from analysis.oceanographic_analyzer import oceanographic_analyzer
from session.session_manager import session_manager, UserQuery, UserPreferences

logger = logging.getLogger(__name__)

# Create router for advanced features
advanced_router = APIRouter(prefix="/advanced", tags=["Advanced Features"])

# Pydantic models for requests
class ExportRequest(BaseModel):
    query: str
    format_type: str  # csv, excel, pdf, json, xml, kml, geojson
    filename: Optional[str] = None
    include_metadata: bool = True
    include_visualizations: bool = False

class AnalysisRequest(BaseModel):
    query: str
    analysis_type: str  # statistical, trend, anomaly, correlation, depth_profile
    parameters: Dict[str, Any] = {}

class PreferencesRequest(BaseModel):
    preferred_units: Optional[str] = None
    default_visualization: Optional[str] = None
    theme: Optional[str] = None
    export_format: Optional[str] = None
    max_query_history: Optional[int] = None
    auto_save_results: Optional[bool] = None
    notification_settings: Optional[Dict[str, bool]] = None

# Session management endpoints
@advanced_router.post("/session/create")
async def create_session(request: Request):
    """Create a new user session."""
    try:
        user_agent = request.headers.get("user-agent")
        client_ip = request.client.host if request.client else None
        
        session_id = session_manager.create_session(user_agent, client_ip)
        
        return {
            "success": True,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.get("/session/{session_id}/info")
async def get_session_info(session_id: str):
    """Get session information and statistics."""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        stats = session_manager.get_session_statistics(session_id)
        preferences = session_manager.get_user_preferences(session_id)
        
        return {
            "success": True,
            "session": {
                "session_id": session.session_id,
                "start_time": session.start_time.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "query_count": session.query_count,
                "is_active": session.is_active
            },
            "statistics": stats,
            "preferences": preferences.__dict__ if preferences else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.get("/session/{session_id}/history")
async def get_query_history(session_id: str, 
                          limit: int = 50,
                          query_type: Optional[str] = None,
                          success_only: bool = False):
    """Get query history for a session."""
    try:
        history = session_manager.get_query_history(
            session_id, limit, query_type, success_only
        )
        
        # Convert to serializable format
        history_data = []
        for query in history:
            history_data.append({
                "query_id": query.query_id,
                "timestamp": query.timestamp.isoformat(),
                "query_text": query.query_text,
                "query_type": query.query_type,
                "parameters": query.parameters,
                "processing_time": query.processing_time,
                "success": query.success,
                "result_summary": query.result_summary,
                "error_message": query.error_message
            })
        
        return {
            "success": True,
            "session_id": session_id,
            "history": history_data,
            "total_queries": len(history_data)
        }
    except Exception as e:
        logger.error(f"Query history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@advanced_router.put("/session/{session_id}/preferences")
async def update_preferences(session_id: str, preferences: PreferencesRequest):
    """Update user preferences for a session."""
    try:
        current_prefs = session_manager.get_user_preferences(session_id)
        if not current_prefs:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update only provided fields
        if preferences.preferred_units is not None:
            current_prefs.preferred_units = preferences.preferred_units
        if preferences.default_visualization is not None:
            current_prefs.default_visualization = preferences.default_visualization
        if preferences.theme is not None:
            current_prefs.theme = preferences.theme
        if preferences.export_format is not None:
            current_prefs.export_format = preferences.export_format
        if preferences.max_query_history is not None:
            current_prefs.max_query_history = preferences.max_query_history
        if preferences.auto_save_results is not None:
            current_prefs.auto_save_results = preferences.auto_save_results
        if preferences.notification_settings is not None:
            current_prefs.notification_settings.update(preferences.notification_settings)
        
        success = session_manager.update_user_preferences(session_id, current_prefs)
        
        if success:
            return {
                "success": True,
                "message": "Preferences updated successfully",
                "preferences": current_prefs.__dict__
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preferences update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced export endpoints
@advanced_router.post("/export")
async def export_data(export_request: ExportRequest, request: Request):
    """Export data in various formats."""
    try:
        # Get session ID from headers or create anonymous session
        session_id = request.headers.get("x-session-id")
        if not session_id:
            session_id = session_manager.create_session()
        
        start_time = datetime.now()
        
        # This would integrate with your existing data retrieval logic
        # For now, we'll simulate data retrieval
        # In a real implementation, you'd call your RAG pipeline here
        simulated_data = [
            {"latitude": 45.0, "longitude": -120.0, "temperature": 15.5, "salinity": 35.2, "depth": 10},
            {"latitude": 46.0, "longitude": -121.0, "temperature": 14.8, "salinity": 35.0, "depth": 50},
            {"latitude": 47.0, "longitude": -122.0, "temperature": 16.2, "salinity": 35.5, "depth": 100}
        ]
        
        metadata = {
            "query": export_request.query,
            "export_timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "total_records": len(simulated_data)
        }
        
        # Export data
        exported_data = advanced_exporter.export_data(
            data=simulated_data,
            format_type=export_request.format_type,
            filename=export_request.filename,
            metadata=metadata if export_request.include_metadata else None
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Record query in session
        query_record = UserQuery(
            query_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=start_time,
            query_text=export_request.query,
            query_type="export",
            parameters={"format": export_request.format_type},
            processing_time=processing_time,
            success=True,
            result_summary=f"Exported {len(simulated_data)} records as {export_request.format_type}"
        )
        session_manager.record_query(query_record)
        
        # Return appropriate response based on format
        if export_request.format_type in ['excel', 'pdf']:
            # Binary formats
            filename = export_request.filename or f"argo_export.{export_request.format_type}"
            return StreamingResponse(
                BytesIO(exported_data),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            # Text formats
            return {
                "success": True,
                "format": export_request.format_type,
                "data": exported_data,
                "metadata": metadata,
                "processing_time": processing_time
            }
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        
        # Record failed query
        if 'session_id' in locals():
            error_query = UserQuery(
                query_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=start_time,
                query_text=export_request.query,
                query_type="export",
                parameters={"format": export_request.format_type},
                processing_time=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_message=str(e)
            )
            session_manager.record_query(error_query)
        
        raise HTTPException(status_code=500, detail=str(e))

# Advanced analysis endpoints
@advanced_router.post("/analysis")
async def perform_analysis(analysis_request: AnalysisRequest, request: Request):
    """Perform advanced oceanographic analysis."""
    try:
        # Get session ID
        session_id = request.headers.get("x-session-id")
        if not session_id:
            session_id = session_manager.create_session()
        
        start_time = datetime.now()
        
        # Simulate data retrieval (replace with actual data fetching)
        # In real implementation, this would query your database based on the request
        import numpy as np
        import pandas as pd
        
        np.random.seed(42)
        depths = np.linspace(0, 2000, 100)
        temperatures = 25 - depths/100 + np.random.normal(0, 1, 100)
        salinities = 34 + depths/1000 + np.random.normal(0, 0.2, 100)
        
        data = pd.DataFrame({
            'depth': depths,
            'temperature': temperatures,
            'salinity': salinities,
            'date': pd.date_range('2024-01-01', periods=100, freq='D')
        })
        
        # Perform analysis based on type
        analysis_type = analysis_request.analysis_type
        parameters = analysis_request.parameters
        
        if analysis_type == "statistical":
            result = oceanographic_analyzer.statistical_summary(
                data, parameters.get("parameters")
            )
        elif analysis_type == "correlation":
            result = oceanographic_analyzer.correlation_analysis(
                data, parameters.get("parameters"), parameters.get("method", "pearson")
            )
        elif analysis_type == "anomaly":
            result = oceanographic_analyzer.anomaly_detection(
                data, parameters.get("parameters", ["temperature", "salinity"]),
                parameters.get("method", "isolation_forest"),
                parameters.get("contamination", 0.1)
            )
        elif analysis_type == "trend":
            result = oceanographic_analyzer.trend_analysis(
                data, parameters.get("time_column", "date"),
                parameters.get("value_column", "temperature"),
                parameters.get("method", "linear")
            )
        elif analysis_type == "depth_profile":
            result = oceanographic_analyzer.depth_profile_analysis(
                data, parameters.get("depth_column", "depth"),
                parameters.get("parameter_column", "temperature")
            )
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Record successful query
        query_record = UserQuery(
            query_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=start_time,
            query_text=analysis_request.query,
            query_type="analysis",
            parameters={"analysis_type": analysis_type, **parameters},
            processing_time=processing_time,
            success=True,
            result_summary=f"Completed {analysis_type} analysis"
        )
        session_manager.record_query(query_record)
        
        return {
            "success": True,
            "analysis_type": analysis_type,
            "query": analysis_request.query,
            "result": result,
            "processing_time": processing_time,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        
        # Record failed query
        if 'session_id' in locals():
            error_query = UserQuery(
                query_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=start_time,
                query_text=analysis_request.query,
                query_type="analysis",
                parameters={"analysis_type": analysis_request.analysis_type},
                processing_time=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_message=str(e)
            )
            session_manager.record_query(error_query)
        
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@advanced_router.get("/capabilities")
async def get_capabilities():
    """Get information about available advanced features."""
    return {
        "success": True,
        "capabilities": {
            "export_formats": advanced_exporter.supported_formats,
            "analysis_types": [
                "statistical", "correlation", "anomaly", "trend", "depth_profile"
            ],
            "session_management": {
                "query_history": True,
                "user_preferences": True,
                "session_statistics": True
            },
            "features": {
                "multi_format_export": True,
                "statistical_analysis": True,
                "anomaly_detection": True,
                "trend_analysis": True,
                "correlation_analysis": True,
                "depth_profile_analysis": True,
                "session_tracking": True,
                "user_preferences": True
            }
        }
    }

@advanced_router.get("/health")
async def advanced_health_check():
    """Health check for advanced features."""
    try:
        # Test each component
        test_results = {
            "exporter": "ok",
            "analyzer": "ok", 
            "session_manager": "ok"
        }
        
        # Test exporter
        try:
            test_data = [{"test": "data"}]
            advanced_exporter.export_data(test_data, 'json')
        except Exception as e:
            test_results["exporter"] = f"error: {str(e)}"
        
        # Test analyzer
        try:
            import pandas as pd
            test_df = pd.DataFrame({"temp": [20, 21, 22], "depth": [0, 10, 20]})
            oceanographic_analyzer.statistical_summary(test_df)
        except Exception as e:
            test_results["analyzer"] = f"error: {str(e)}"
        
        # Test session manager
        try:
            test_session = session_manager.create_session()
            session_manager.get_session(test_session)
        except Exception as e:
            test_results["session_manager"] = f"error: {str(e)}"
        
        all_ok = all(result == "ok" for result in test_results.values())
        
        return {
            "success": all_ok,
            "status": "healthy" if all_ok else "degraded",
            "components": test_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Export the router for integration with main API
__all__ = ['advanced_router']