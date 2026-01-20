"""
Dynamic Visualization Engine for ARGO Ocean Data Explorer

Main orchestration module that combines:
- Visualization Intelligence (LLM-based chart type selection)
- Chart Generation Engine (dynamic Plotly/Matplotlib charts)
- RAG Pipeline Integration (enhanced context for visualizations)
- MCP Server Integration (data retrieval and analysis tools)

This creates a seamless query-to-visualization workflow.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import pandas as pd
from pathlib import Path

# Local imports
from .intelligence import visualization_intelligence
from .chart_generator import chart_generator
from rag.rag_pipeline import ArgoRAGPipeline
from config import config

logger = logging.getLogger(__name__)

class DynamicVisualizationEngine:
    """
    Main visualization engine that orchestrates the complete query-to-visualization pipeline
    """
    
    def __init__(self):
        self.rag_pipeline = None
        self.mcp_tools = {}
        self.visualization_cache = {}
        self.cache_file = Path("data/visualization_cache.json")
        self.max_cache_size = 100
        
        # Initialize components
        self.intelligence = visualization_intelligence
        self.chart_generator = chart_generator
        
        logger.info("Dynamic Visualization Engine initialized")
    
    async def initialize(self):
        """Initialize the visualization engine with all dependencies"""
        try:
            # Initialize RAG pipeline
            self.rag_pipeline = ArgoRAGPipeline()
            await self.rag_pipeline.initialize_postgresql()
            
            # Load MCP tools (simplified - in practice would import from MCP server)
            self._load_mcp_tools()
            
            # Load visualization cache
            self._load_visualization_cache()
            
            logger.info("Dynamic Visualization Engine fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize visualization engine: {str(e)}")
            return False
    
    def _load_mcp_tools(self):
        """Load simplified MCP tool references"""
        # This would integrate with the actual MCP server in practice
        self.mcp_tools = {
            "get_temperature_profiles": "Retrieve temperature profile data",
            "get_geographic_distribution": "Get geographic distribution data",
            "get_temporal_analysis": "Perform temporal analysis",
            "get_water_mass_analysis": "Analyze water mass properties",
            "get_regional_statistics": "Calculate regional statistics"
        }
        logger.info(f"Loaded {len(self.mcp_tools)} MCP tool references")
    
    def _load_visualization_cache(self):
        """Load visualization cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.visualization_cache = json.load(f)
                logger.info(f"Loaded {len(self.visualization_cache)} cached visualizations")
            except Exception as e:
                logger.warning(f"Failed to load visualization cache: {str(e)}")
                self.visualization_cache = {}
        else:
            self.visualization_cache = {}
    
    def _save_visualization_cache(self):
        """Save visualization cache to disk"""
        try:
            # Limit cache size
            if len(self.visualization_cache) > self.max_cache_size:
                # Remove oldest entries
                sorted_items = sorted(
                    self.visualization_cache.items(), 
                    key=lambda x: x[1].get('timestamp', ''),
                    reverse=True
                )
                self.visualization_cache = dict(sorted_items[:self.max_cache_size])
            
            self.cache_file.parent.mkdir(exist_ok=True)
            with open(self.cache_file, 'w') as f:
                # Only save metadata, not actual chart data
                cache_metadata = {}
                for key, value in self.visualization_cache.items():
                    cache_metadata[key] = {
                        'timestamp': value.get('timestamp'),
                        'query': value.get('query'),
                        'visualization_type': value.get('visualization_type'),
                        'data_summary': value.get('data_summary', {})
                    }
                json.dump(cache_metadata, f, indent=2)
            
        except Exception as e:
            logger.warning(f"Failed to save visualization cache: {str(e)}")
    
    async def create_visualization(self, 
                                 query: str,
                                 output_format: str = "plotly",
                                 theme: str = "scientific",
                                 use_cache: bool = True) -> Dict[str, Any]:
        """
        Main method to create visualization from natural language query
        
        Args:
            query: Natural language query about oceanographic data
            output_format: "plotly", "matplotlib", or "both"
            theme: "scientific", "presentation", or "dark"
            use_cache: Whether to use cached results
            
        Returns:
            Complete visualization result with charts, metadata, and analysis
        """
        try:
            # Check cache first
            cache_key = f"{hash(query)}_{output_format}_{theme}"
            if use_cache and cache_key in self.visualization_cache:
                logger.info(f"Returning cached visualization for query: {query[:50]}...")
                return self.visualization_cache[cache_key]
            
            logger.info(f"Creating visualization for query: '{query}'")
            
            # Step 1: Get enhanced context from RAG pipeline
            enhanced_context = await self._get_enhanced_context(query)
            
            # Step 2: Retrieve relevant data using MCP tools
            data_context, retrieved_data = await self._retrieve_visualization_data(query, enhanced_context)
            
            # Step 3: Analyze query for optimal visualization strategy
            visualization_analysis = await self.intelligence.analyze_query_for_visualization(
                query, data_context
            )
            
            # Step 4: Generate visualization
            chart_result = await self.chart_generator.generate_chart(
                visualization_analysis, retrieved_data, output_format, theme
            )
            
            # Step 5: Create comprehensive result
            result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "enhanced_context": enhanced_context,
                "visualization_analysis": visualization_analysis,
                "data_context": data_context,
                "chart_result": chart_result,
                "success": True,
                "metadata": {
                    "engine_version": "1.0",
                    "processing_time": "calculated_separately",
                    "data_points": len(retrieved_data) if isinstance(retrieved_data, pd.DataFrame) else 0,
                    "query_complexity": self._assess_query_complexity(query),
                    "recommendations": self._generate_recommendations(visualization_analysis, data_context)
                }
            }
            
            # Cache result (without large chart data)
            if use_cache:
                self.visualization_cache[cache_key] = result
                self._save_visualization_cache()
            
            logger.info(f"Successfully created {visualization_analysis['primary_visualization']} visualization")
            return result
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            return {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e),
                "fallback_suggestion": "Try a simpler query like 'Show temperature profiles' or 'Map float locations'"
            }
    
    async def _get_enhanced_context(self, query: str) -> str:
        """Get enhanced context from RAG pipeline"""
        try:
            if self.rag_pipeline:
                context = await self.rag_pipeline.get_enhanced_context_for_llm(query, top_k=5)
                return context
            else:
                return f"Enhanced context not available for query: {query}"
        except Exception as e:
            logger.warning(f"Failed to get enhanced context: {str(e)}")
            return f"Basic context for visualization query: {query}"
    
    async def _retrieve_visualization_data(self, query: str, enhanced_context: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Retrieve data for visualization using RAG and simulated MCP tools"""
        try:
            # In practice, this would use the actual MCP server tools
            # For now, we'll simulate data retrieval based on query analysis
            
            # Analyze query to determine data needs
            query_lower = query.lower()
            
            if "temperature" in query_lower and "profile" in query_lower:
                # Simulate temperature profile data
                data = self._simulate_temperature_profile_data()
                data_context = {
                    "columns": list(data.columns),
                    "shape": {"rows": len(data), "columns": len(data.columns)},
                    "temporal_range": self._get_temporal_range(data),
                    "spatial_range": self._get_spatial_range(data),
                    "depth_range": self._get_depth_range_context(data),
                    "quality_info": {"total_profiles": 5, "quality_controlled": True}
                }
                
            elif "geographic" in query_lower or "map" in query_lower:
                # Simulate geographic distribution data
                data = self._simulate_geographic_data()
                data_context = {
                    "columns": list(data.columns),
                    "shape": {"rows": len(data), "columns": len(data.columns)},
                    "spatial_range": self._get_spatial_range(data),
                    "quality_info": {"total_floats": len(data), "geographic_coverage": "Indian Ocean"}
                }
                
            elif "time" in query_lower or "temporal" in query_lower:
                # Simulate time series data
                data = self._simulate_time_series_data()
                data_context = {
                    "columns": list(data.columns),
                    "shape": {"rows": len(data), "columns": len(data.columns)},
                    "temporal_range": self._get_temporal_range(data),
                    "quality_info": {"time_span_years": 3, "regular_sampling": False}
                }
                
            else:
                # Default: temperature profile data
                data = self._simulate_temperature_profile_data()
                data_context = {
                    "columns": list(data.columns),
                    "shape": {"rows": len(data), "columns": len(data.columns)},
                    "quality_info": {"default_dataset": True}
                }
            
            logger.info(f"Retrieved {len(data)} data points for visualization")
            return data_context, data
            
        except Exception as e:
            logger.error(f"Data retrieval failed: {str(e)}")
            # Return minimal fallback data
            fallback_data = pd.DataFrame({
                "depth_m": [0, 10, 20, 50, 100],
                "temperature_c": [28.5, 28.0, 27.5, 26.0, 24.0],
                "salinity_psu": [35.0, 35.1, 35.2, 35.3, 35.4]
            })
            return {"columns": list(fallback_data.columns)}, fallback_data
    
    def _simulate_temperature_profile_data(self) -> pd.DataFrame:
        """Simulate realistic temperature profile data"""
        import numpy as np
        
        depths = np.arange(0, 2000, 10)  # 0 to 2000m, every 10m
        n_profiles = 5
        
        data_list = []
        for profile_id in range(1, n_profiles + 1):
            # Simulate realistic temperature profile
            surface_temp = 28 + np.random.normal(0, 1)
            thermocline_depth = 100 + np.random.normal(0, 20)
            deep_temp = 4 + np.random.normal(0, 0.5)
            
            for depth in depths:
                if depth < thermocline_depth:
                    # Mixed layer
                    temp = surface_temp - (depth / thermocline_depth) * 5
                else:
                    # Below thermocline
                    temp = surface_temp - 5 - (depth - thermocline_depth) / 200
                    temp = max(temp, deep_temp)
                
                salinity = 35.0 + depth / 10000 + np.random.normal(0, 0.1)
                
                data_list.append({
                    "profile_id": profile_id,
                    "depth_m": depth,
                    "temperature_c": temp + np.random.normal(0, 0.2),
                    "salinity_psu": salinity,
                    "latitude": -10 + np.random.normal(0, 1),
                    "longitude": 45 + np.random.normal(0, 2),
                    "profile_datetime": pd.Timestamp("2023-01-01") + pd.Timedelta(days=profile_id*30)
                })
        
        return pd.DataFrame(data_list)
    
    def _simulate_geographic_data(self) -> pd.DataFrame:
        """Simulate geographic distribution data"""
        import numpy as np
        
        n_points = 50
        data_list = []
        
        for i in range(n_points):
            data_list.append({
                "latitude": -15 + np.random.normal(0, 3),
                "longitude": 40 + np.random.normal(0, 8),
                "temperature_c": 25 + np.random.normal(0, 3),
                "salinity_psu": 35 + np.random.normal(0, 0.5),
                "float_id": f"FLOAT_{i+1:03d}"
            })
        
        return pd.DataFrame(data_list)
    
    def _simulate_time_series_data(self) -> pd.DataFrame:
        """Simulate time series data"""
        import numpy as np
        
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="M")
        data_list = []
        
        for date in dates:
            # Seasonal temperature variation
            day_of_year = date.dayofyear
            seasonal_temp = 26 + 3 * np.sin(2 * np.pi * day_of_year / 365)
            
            data_list.append({
                "profile_datetime": date,
                "temperature_c": seasonal_temp + np.random.normal(0, 1),
                "salinity_psu": 35.0 + np.random.normal(0, 0.2),
                "depth_m": 50,  # Surface layer
                "latitude": -10,
                "longitude": 45
            })
        
        return pd.DataFrame(data_list)
    
    def _get_temporal_range(self, data: pd.DataFrame) -> Dict[str, str]:
        """Get temporal range from data"""
        if "profile_datetime" in data.columns:
            return {
                "start": data["profile_datetime"].min().isoformat(),
                "end": data["profile_datetime"].max().isoformat()
            }
        return {}
    
    def _get_spatial_range(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get spatial range from data"""
        spatial_range = {}
        if "latitude" in data.columns:
            spatial_range.update({
                "lat_min": float(data["latitude"].min()),
                "lat_max": float(data["latitude"].max())
            })
        if "longitude" in data.columns:
            spatial_range.update({
                "lon_min": float(data["longitude"].min()),
                "lon_max": float(data["longitude"].max())
            })
        return spatial_range
    
    def _get_depth_range_context(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get depth range context"""
        if "depth_m" in data.columns:
            return {
                "min_depth": float(data["depth_m"].min()),
                "max_depth": float(data["depth_m"].max()),
                "depth_resolution": float(data["depth_m"].nunique())
            }
        return {}
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess the complexity of the user query"""
        query_words = query.split()
        
        if len(query_words) <= 5:
            return "simple"
        elif len(query_words) <= 15:
            return "moderate"
        else:
            return "complex"
    
    def _generate_recommendations(self, visualization_analysis: Dict[str, Any], 
                                data_context: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving the visualization"""
        recommendations = []
        
        # Check data quality
        data_shape = data_context.get("shape", {})
        if data_shape.get("rows", 0) < 100:
            recommendations.append("Consider requesting more data points for better statistical analysis")
        
        # Check visualization type appropriateness
        viz_type = visualization_analysis.get("primary_visualization")
        if viz_type == "geographic_map" and "spatial_range" not in data_context:
            recommendations.append("Geographic visualization requires latitude/longitude data")
        
        # Suggest complementary visualizations
        if viz_type == "temperature_profile":
            recommendations.append("Consider also viewing a T-S diagram for water mass analysis")
        elif viz_type == "time_series":
            recommendations.append("A geographic map could show spatial patterns alongside temporal trends")
        
        return recommendations

# Global instance
dynamic_visualization_engine = DynamicVisualizationEngine()