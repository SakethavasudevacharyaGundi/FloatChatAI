#!/usr/bin/env python3
"""
Enhanced Gemini Orchestrator with API Key Management
Handles quota limits gracefully and provides intelligent responses
"""

import os
import json
import logging
import pandas as pd
import google.generativeai as genai
from typing import Dict, Any, List, Optional
import plotly.express as px
import base64
from io import BytesIO

# Import our enhanced components
from api.api_key_manager import api_key_manager
from api.enhanced_gemini_processor import enhanced_gemini_processor

# Assuming these exist and are functional
from rag.rag_pipeline import ArgoRAGPipeline
from mcp.mcp_server import ArgoMCPServer

logger = logging.getLogger(__name__)

class EnhancedGeminiOrchestrator:
    """
    Enhanced orchestrator with API key management and quota handling
    """
    
    def __init__(self, data_path: str = 'data/1900121_prof.csv'):
        self.data = self._load_data(data_path)
        self.rag_pipeline = None
        self.mcp_server = None
        self.is_initialized = False
        
        # Initialize components
        self._initialize_components()
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load ARGO data from CSV"""
        try:
            data = pd.read_csv(data_path)
            logger.info(f"✅ Loaded {len(data)} ARGO measurements")
            return data
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return pd.DataFrame()
    
    def _initialize_components(self):
        """Initialize RAG and MCP components"""
        try:
            # Initialize RAG pipeline
            self.rag_pipeline = ArgoRAGPipeline()
            logger.info(f"✅ RAG Pipeline initialized with {len(self.rag_pipeline.documents)} documents")
            
            # Initialize MCP server
            self.mcp_server = ArgoMCPServer()
            logger.info(f"✅ MCP Server initialized with {len(self.mcp_server.tools)} oceanographic tools")
            
            self.is_initialized = True
            logger.info("🎉 Enhanced orchestrator fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            self.is_initialized = False
    
    def _get_scientific_context(self, query: str) -> List[str]:
        """Get scientific context from RAG pipeline"""
        if not self.rag_pipeline:
            return []
        
        try:
            context = self.rag_pipeline.retrieve_context(query, top_k=3)
            return [doc.page_content for doc in context]
        except Exception as e:
            logger.error(f"❌ RAG context retrieval failed: {e}")
            return []
    
    def _execute_mcp_task(self, task_name: str, params: Dict[str, Any]) -> Any:
        """Execute MCP task"""
        if not self.mcp_server:
            return None
        
        try:
            result = self.mcp_server.execute_tool(task_name, params)
            return result
        except Exception as e:
            logger.error(f"❌ MCP task execution failed: {e}")
            return None
    
    def _generate_visualization(self, data: pd.DataFrame, request: str) -> Optional[str]:
        """Generate visualization based on request"""
        try:
            if "temperature" in request.lower() and "depth" in request.lower():
                fig = px.scatter(data, x='temperature', y='pressure', 
                               color='salinity', title='Temperature vs Depth')
                fig.update_layout(yaxis=dict(autorange="reversed"))
            elif "spatial" in request.lower() or "map" in request.lower():
                fig = px.scatter(data, x='longitude', y='latitude', 
                               color='temperature', title='Spatial Distribution')
            else:
                fig = px.scatter(data, x='temperature', y='salinity', 
                               title='Temperature vs Salinity')
            
            # Convert to base64
            img_bytes = fig.to_image(format="png", width=800, height=600)
            img_base64 = base64.b64encode(img_bytes).decode()
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"❌ Visualization generation failed: {e}")
            return None
    
    def _execute_real_data_queries(self, query: str) -> Dict[str, Any]:
        """Execute real data queries based on user input"""
        try:
            analysis = {}
            
            # Basic statistics
            analysis['total_measurements'] = len(self.data)
            analysis['unique_floats'] = self.data['profile_index'].nunique()
            analysis['measurements_per_float'] = len(self.data) / self.data['profile_index'].nunique()
            
            # Geographic coverage
            analysis['geographic_coverage'] = {
                'latitude_range': [self.data['latitude'].min(), self.data['latitude'].max()],
                'longitude_range': [self.data['longitude'].min(), self.data['longitude'].max()]
            }
            
            # Parameter ranges
            analysis['parameter_ranges'] = {
                'temperature': [self.data['temperature'].min(), self.data['temperature'].max()],
                'salinity': [self.data['salinity'].min(), self.data['salinity'].max()],
                'pressure': [self.data['pressure'].min(), self.data['pressure'].max()]
            }
            
            # Query-specific analysis
            if "temperature" in query.lower():
                analysis['temperature_analysis'] = {
                    'average': self.data['temperature'].mean(),
                    'std': self.data['temperature'].std(),
                    'min': self.data['temperature'].min(),
                    'max': self.data['temperature'].max()
                }
            
            if "salinity" in query.lower():
                analysis['salinity_analysis'] = {
                    'average': self.data['salinity'].mean(),
                    'std': self.data['salinity'].std(),
                    'min': self.data['salinity'].min(),
                    'max': self.data['salinity'].max()
                }
            
            if "spatial" in query.lower() or "location" in query.lower():
                analysis['spatial_analysis'] = {
                    'total_locations': self.data['profile_index'].nunique(),
                    'latitude_center': self.data['latitude'].mean(),
                    'longitude_center': self.data['longitude'].mean(),
                    'geographic_spread': {
                        'lat_range': self.data['latitude'].max() - self.data['latitude'].min(),
                        'lon_range': self.data['longitude'].max() - self.data['longitude'].min()
                    }
                }
            
            # Summary
            analysis['summary'] = f"Analysis of {len(self.data)} ARGO measurements from {self.data['profile_index'].nunique()} floats in the Indian Ocean region."
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Real data query execution failed: {e}")
            return {'error': str(e)}
    
    async def chat(self, user_query: str) -> Dict[str, Any]:
        """
        Enhanced chat with API key management and quota handling
        """
        logger.info(f"🔍 Processing query: {user_query}")
        
        try:
            # Step 1: Execute real data queries
            real_analysis = self._execute_real_data_queries(user_query)
            
            # Step 2: Get scientific context
            rag_context = self._get_scientific_context(user_query)
            
            # Step 3: Generate visualizations
            visualizations = []
            if "temperature" in user_query.lower() or "depth" in user_query.lower():
                viz = self._generate_visualization(self.data, "temperature depth")
                if viz:
                    visualizations.append(viz)
            
            if "spatial" in user_query.lower() or "map" in user_query.lower():
                viz = self._generate_visualization(self.data, "spatial")
                if viz:
                    visualizations.append(viz)
            
            # Step 4: Use enhanced Gemini processor
            response_data = enhanced_gemini_processor.process_query(
                query=user_query,
                data=self.data,
                context=rag_context,
                visualizations=visualizations,
                real_analysis=real_analysis
            )
            
            # Add additional metadata
            response_data.update({
                "rag_context": rag_context,
                "real_analysis": real_analysis,
                "visualizations": visualizations,
                "data_summary": {
                    "total_measurements": len(self.data),
                    "unique_floats": self.data['profile_index'].nunique(),
                    "geographic_coverage": real_analysis.get('geographic_coverage', {}),
                    "parameter_ranges": real_analysis.get('parameter_ranges', {})
                }
            })
            
            logger.info("✅ Enhanced response generated successfully")
            return response_data
            
        except Exception as e:
            logger.error(f"❌ Enhanced chat processing failed: {e}")
            return {
                "response": f"I apologize, but I'm experiencing technical difficulties. Error: {str(e)}",
                "source": "error",
                "status": "error",
                "timestamp": pd.Timestamp.now().isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "is_initialized": self.is_initialized,
            "data_loaded": len(self.data) > 0,
            "rag_available": self.rag_pipeline is not None,
            "mcp_available": self.mcp_server is not None,
            "gemini_status": enhanced_gemini_processor.get_status(),
            "api_key_status": api_key_manager.get_status()
        }

# Global instance
enhanced_orchestrator = EnhancedGeminiOrchestrator()
