"""
Data models for FloatChatAI - ARGO Ocean Data Explorer
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ArgoFloat(BaseModel):
    """ARGO Float data model"""
    float_id: str
    platform_number: Optional[str] = None
    wmo_number: Optional[str] = None
    deployment_date: Optional[datetime] = None
    status: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "float_id": "1900121",
                "platform_number": "1900121",
                "wmo_number": "1900121",
                "deployment_date": "2006-01-01T00:00:00",
                "status": "active",
                "country": "US",
                "latitude": 45.5,
                "longitude": -120.3
            }
        }


class ArgoProfile(BaseModel):
    """ARGO Profile data model"""
    profile_id: Optional[str] = None
    float_id: str
    cycle_number: Optional[int] = None
    date: Optional[datetime] = None
    latitude: float
    longitude: float
    temperature: Optional[List[float]] = None
    salinity: Optional[List[float]] = None
    pressure: Optional[List[float]] = None
    depth: Optional[List[float]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "profile_id": "1900121_001",
                "float_id": "1900121",
                "cycle_number": 1,
                "date": "2006-01-15T12:00:00",
                "latitude": 45.5,
                "longitude": -120.3,
                "temperature": [15.2, 14.8, 14.5],
                "salinity": [35.1, 35.2, 35.3],
                "pressure": [10, 20, 30],
                "depth": [10, 20, 30]
            }
        }


class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., description="Natural language query about ARGO ocean data")
    include_visualization: bool = Field(default=True, description="Whether to include visualizations")
    limit: Optional[int] = Field(default=50, description="Maximum number of results to return")
    float_id: Optional[str] = Field(default=None, description="Specific float ID to query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Show me temperature profiles for float 1900121",
                "include_visualization": True,
                "limit": 50,
                "float_id": "1900121"
            }
        }


class QueryResponse(BaseModel):
    """Query response model"""
    query: str
    response: str
    data: Optional[Dict[str, Any]] = None
    visualization: Optional[Dict[str, Any]] = None
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Show me temperature profiles for float 1900121",
                "response": "Found 42 temperature profiles for float 1900121...",
                "data": {"profiles": []},
                "visualization": {"type": "plotly", "data": {}},
                "processing_time": 0.523,
                "metadata": {"source": "postgresql", "count": 42}
            }
        }
