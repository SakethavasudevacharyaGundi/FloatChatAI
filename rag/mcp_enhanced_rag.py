"""
MCP-Enhanced RAG Pipeline for ARGO Ocean Data System
Integrates Model Context Protocol for enhanced AI query processing
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import json
import re
import pandas as pd
from datetime import datetime, timedelta
import logging

from vector_db.vector_store import ArgoVectorStore
from rag.query_processor import ArgoQueryProcessor
from models.data_models import QueryRequest, QueryResponse
from config import config
from mcp.mcp_client import MCPToolManager, get_mcp_context_for_gemini, process_query_with_mcp

logger = logging.getLogger(__name__)

class EnhancedArgoRAGPipeline:
    """Enhanced RAG pipeline with MCP integration for intelligent ARGO data analysis"""
    
    def __init__(self):
        self.vector_store = ArgoVectorStore(config.CHROMA_PERSIST_DIRECTORY)
        self.query_processor = ArgoQueryProcessor()
        self.mcp_manager = MCPToolManager()
        self.mcp_enabled = False
        
        # Initialize LLM with enhanced context
        self._initialize_llm()
        
        # Schema information for SQL generation
        self.schema_info = self._get_enhanced_schema_info()
        
        # Enhanced prompt templates
        self.enhanced_prompt = self._create_enhanced_prompt()
        
    def _initialize_llm(self):
        """Initialize Gemini LLM with enhanced configuration"""
        try:
            if config.GOOGLE_API_KEY:
                genai.configure(api_key=config.GOOGLE_API_KEY)
                
                self.llm = ChatGoogleGenerativeAI(
                    model=config.GEMINI_MODEL,
                    temperature=0.1,
                    max_output_tokens=2000,
                    google_api_key=config.GOOGLE_API_KEY
                )
                
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=config.GEMINI_EMBEDDING_MODEL,
                    google_api_key=config.GOOGLE_API_KEY
                )
                
                logger.info("Gemini LLM initialized successfully")
            else:
                raise ValueError("No Google API key provided")
        except Exception as e:
            logger.error(f"Error initializing Gemini: {e}")
            self.llm = None
            self.embeddings = None
    
    async def initialize(self):
        """Initialize the enhanced RAG pipeline with MCP"""
        try:
            # Initialize MCP manager
            mcp_success = await self.mcp_manager.initialize()
            if mcp_success:
                self.mcp_enabled = True
                logger.info("MCP integration enabled in RAG pipeline")
            else:
                logger.warning("MCP initialization failed, continuing without MCP")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize enhanced RAG pipeline: {e}")
            return False
    
    def _get_enhanced_schema_info(self) -> str:
        """Get enhanced database schema information with MCP context"""
        return """
        🌊 ARGO Oceanographic Database Schema:
        
        📊 Table: argo_profiles (Oceanographic measurements)
        Essential Columns:
        - profile_id (TEXT): Unique measurement profile identifier
        - float_id (TEXT): ARGO float instrument identifier  
        - latitude (REAL): Geographic latitude (-90 to 90)
        - longitude (REAL): Geographic longitude (-180 to 180)
        - timestamp (TEXT): Measurement date/time (ISO format)
        - depth (REAL): Measurement depth in meters (0-6000)
        - temperature (REAL): Water temperature in Celsius
        - salinity (REAL): Salinity in PSU (Practical Salinity Units)
        - pressure (REAL): Water pressure in decibars
        - oxygen (REAL): Dissolved oxygen (μmol/kg) [BGC parameter]
        - chlorophyll (REAL): Chlorophyll-a concentration (mg/m³) [BGC parameter]
        - nitrate (REAL): Nitrate concentration (μmol/kg) [BGC parameter]
        - ph (REAL): pH measurement [BGC parameter]
        - region (TEXT): Ocean region classification
        - quality_flag (INTEGER): Data quality (1=excellent, 2=good, 3=questionable, 4=bad)
        
        🤖 Table: argo_floats (Instrument information)  
        Essential Columns:
        - float_id (TEXT): Unique float identifier
        - wmo_id (TEXT): World Meteorological Organization ID
        - status (TEXT): Operational status (active/inactive/lost)
        - deployment_date (TEXT): Deployment timestamp
        - last_profile_date (TEXT): Most recent measurement
        - total_profiles (INTEGER): Total measurements collected
        - region (TEXT): Primary operational region
        - latitude/longitude (REAL): Current position
        
        🔍 Enhanced Query Capabilities:
        1. Natural Language → SQL conversion
        2. MCP tool integration for complex queries
        3. Semantic search across oceanographic data
        4. Intelligent visualization recommendations
        5. Multi-modal data analysis (Core + BGC parameters)
        """
    
    def _create_enhanced_prompt(self) -> str:
        """Create enhanced prompt template with MCP integration"""
        return """
        🌊 You are an expert ARGO Oceanographic Data Assistant with advanced AI capabilities.
        
        🧠 Available Analysis Tools:
        {mcp_context}
        
        📋 Database Schema:
        {schema_info}
        
        🎯 Your Mission:
        1. Understand complex oceanographic queries in natural language
        2. Use MCP tools when available for enhanced data access
        3. Generate precise SQL queries for direct database access
        4. Provide intelligent insights about ocean data patterns
        5. Recommend appropriate visualizations for data exploration
        6. Explain oceanographic concepts clearly and scientifically
        
        ⚡ Query Processing Strategy:
        - For simple data retrieval: Generate optimized SQL
        - For complex analysis: Use MCP tools + SQL combination  
        - For exploration: Provide semantic search + recommendations
        - For BGC data: Include biogeochemical parameter analysis
        
        📊 Visualization Intelligence:
        - Profile plots: Temperature/salinity vs depth
        - Time series: Parameter evolution over time
        - Spatial maps: Geographic distribution of measurements
        - Correlation plots: Multi-parameter relationships
        - Float tracking: Instrument trajectory analysis
        
        🔬 Scientific Context:
        Always consider:
        - Oceanographic principles (density, mixed layer, thermocline)
        - Seasonal variations and climate patterns
        - Data quality and uncertainty
        - Regional ocean characteristics
        - BGC cycling and marine ecosystem health
        
        Current Query: {query}
        
        Provide a comprehensive response including:
        1. Query interpretation and scientific context
        2. Recommended analysis approach (MCP tools + SQL)
        3. Expected data insights and patterns
        4. Optimal visualization strategy
        5. Follow-up research suggestions
        """
    
    async def process_enhanced_query(self, request: QueryRequest) -> QueryResponse:
        """Process query with enhanced MCP + RAG capabilities"""
        try:
            query = request.query
            logger.info(f"Processing enhanced query: {query}")
            
            # Step 1: Get MCP context if available
            mcp_context = ""
            mcp_results = {}
            
            if self.mcp_enabled:
                mcp_context = await get_mcp_context_for_gemini()
                mcp_results = await process_query_with_mcp(query)
                logger.info(f"MCP tools executed: {list(mcp_results.keys())}")
            
            # Step 2: Enhance query with AI analysis
            if self.llm:
                enhanced_prompt = self.enhanced_prompt.format(
                    mcp_context=mcp_context,
                    schema_info=self.schema_info,
                    query=query
                )
                
                ai_response = self.llm.predict(enhanced_prompt)
                logger.info("AI analysis completed")
            else:
                ai_response = self._rule_based_analysis(query)
            
            # Step 3: Generate SQL query
            sql_query = self._extract_sql_from_response(ai_response, query)
            
            # Step 4: Combine all results
            combined_results = self._combine_results(mcp_results, sql_query, query)
            
            # Step 5: Determine optimal visualization
            viz_type = self._determine_enhanced_visualization(query, combined_results)
            
            # Step 6: Generate comprehensive response
            metadata = {
                "ai_analysis": ai_response,
                "mcp_enabled": self.mcp_enabled,
                "mcp_tools_used": list(mcp_results.keys()) if mcp_results else [],
                "query_classification": self._classify_enhanced_query(query),
                "oceanographic_context": self._get_oceanographic_context(query),
                "visualization_recommendations": self._get_viz_recommendations(query),
                "follow_up_suggestions": self._generate_follow_up_suggestions(query),
                "data_quality_notes": self._get_data_quality_notes(query),
                "scientific_insights": self._extract_scientific_insights(ai_response)
            }
            
            return QueryResponse(
                query=query,
                sql_query=sql_query,
                results=combined_results,
                visualization_type=viz_type,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced query processing: {e}")
            return self._create_error_response(request.query, str(e))
    
    def _combine_results(self, mcp_results: Dict, sql_query: str, query: str) -> List[Dict]:
        """Combine MCP tool results with SQL query results"""
        combined = []
        
        # Add MCP results
        for tool_name, tool_result in mcp_results.items():
            if tool_result.get('success') and tool_result.get('data'):
                if isinstance(tool_result['data'], list):
                    combined.extend(tool_result['data'])
                elif isinstance(tool_result['data'], dict):
                    combined.append(tool_result['data'])
        
        # Add mock SQL results (in production, execute real SQL)
        sql_results = self._execute_enhanced_mock_query(sql_query, query)
        combined.extend(sql_results)
        
        return combined[:100]  # Limit results
    
    def _execute_enhanced_mock_query(self, sql_query: str, query: str) -> List[Dict]:
        """Execute enhanced mock query with realistic ARGO data patterns"""
        query_lower = query.lower()
        
        # Generate realistic oceanographic data based on query type
        if any(word in query_lower for word in ["temperature", "salinity", "profile"]):
            return self._generate_realistic_profile_data(query_lower)
        elif "float" in query_lower:
            return self._generate_realistic_float_data(query_lower)
        elif any(word in query_lower for word in ["oxygen", "chlorophyll", "bgc", "biogeochemical"]):
            return self._generate_realistic_bgc_data(query_lower)
        else:
            return self._generate_realistic_profile_data(query_lower)
    
    def _generate_realistic_profile_data(self, query_context: str) -> List[Dict]:
        """Generate realistic oceanographic profile data"""
        profiles = []
        
        # Determine region-specific characteristics
        if "indian" in query_context:
            base_lat, base_lon = 15.0, 70.0
            temp_offset = 2.0  # Warmer Indian Ocean
        elif "atlantic" in query_context:
            base_lat, base_lon = 25.0, -40.0
            temp_offset = 0.0
        elif "pacific" in query_context:
            base_lat, base_lon = 10.0, 150.0
            temp_offset = 1.0
        else:
            base_lat, base_lon = 0.0, 0.0
            temp_offset = 0.0
        
        for i in range(25):
            depth = i * 50  # Every 50m
            
            # Realistic temperature profile (warmer at surface, cooler with depth)
            temperature = 28.0 + temp_offset - (depth * 0.01) - (depth/200)**2
            
            # Realistic salinity profile (varies with depth and region)
            salinity = 35.0 + 0.2 * (depth/1000) - 0.1 * (depth/1000)**2
            
            # Realistic pressure (approximately 1 dbar per meter)
            pressure = depth * 1.02
            
            profile = {
                "profile_id": f"ENHANCED_PROFILE_{i:03d}",
                "float_id": f"ARGO_ENHANCED_{100000 + i}",
                "latitude": base_lat + (i * 0.01),
                "longitude": base_lon + (i * 0.02),
                "depth": depth,
                "temperature": round(temperature, 2),
                "salinity": round(salinity, 3),
                "pressure": round(pressure, 1),
                "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                "region": "enhanced_region",
                "quality_flag": 1 if i % 5 != 0 else 2
            }
            
            profiles.append(profile)
        
        return profiles
    
    def _generate_realistic_float_data(self, query_context: str) -> List[Dict]:
        """Generate realistic ARGO float data"""
        floats = []
        
        for i in range(12):
            float_data = {
                "float_id": f"ENHANCED_FLOAT_{200000 + i}",
                "wmo_id": f"{2900000 + i}",
                "status": "active" if i % 4 != 0 else "inactive",
                "latitude": 15.0 + (i * 3),
                "longitude": 70.0 + (i * 2.5),
                "region": "indian_ocean" if i % 3 == 0 else "pacific_ocean",
                "deployment_date": (datetime.now() - timedelta(days=365 + i * 45)).isoformat(),
                "last_profile_date": (datetime.now() - timedelta(days=i * 2)).isoformat(),
                "total_profiles": 200 + i * 15,
                "platform_type": "APEX" if i % 2 == 0 else "NAVIS"
            }
            floats.append(float_data)
        
        return floats
    
    def _generate_realistic_bgc_data(self, query_context: str) -> List[Dict]:
        """Generate realistic biogeochemical (BGC) data"""
        bgc_profiles = []
        
        for i in range(20):
            depth = i * 25
            
            # Realistic BGC parameters
            oxygen = max(200 - depth * 0.5, 50)  # Oxygen decreases with depth
            chlorophyll = max(2.0 - depth * 0.01, 0.1) if depth < 150 else 0.1  # Chlorophyll max near surface
            nitrate = min(depth * 0.02, 30)  # Nitrate increases with depth
            ph = 8.1 - depth * 0.0001  # pH slightly decreases with depth
            
            bgc_profile = {
                "profile_id": f"BGC_PROFILE_{i:03d}",
                "float_id": f"BGC_FLOAT_{300000 + i}",
                "latitude": 10.0 + (i * 0.5),
                "longitude": 65.0 + (i * 0.3),
                "depth": depth,
                "oxygen": round(oxygen, 1),
                "chlorophyll": round(chlorophyll, 2),
                "nitrate": round(nitrate, 1),
                "ph": round(ph, 2),
                "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                "region": "tropical_ocean",
                "quality_flag": 1
            }
            bgc_profiles.append(bgc_profile)
        
        return bgc_profiles
    
    def _classify_enhanced_query(self, query: str) -> str:
        """Enhanced query classification with MCP awareness"""
        query_lower = query.lower()
        
        classifications = []
        
        # Data type classification
        if any(word in query_lower for word in ["temperature", "temp"]):
            classifications.append("temperature_analysis")
        if any(word in query_lower for word in ["salinity", "salt"]):
            classifications.append("salinity_analysis")
        if any(word in query_lower for word in ["oxygen", "o2"]):
            classifications.append("bgc_oxygen")
        if any(word in query_lower for word in ["chlorophyll", "chl"]):
            classifications.append("bgc_chlorophyll")
        
        # Analysis type classification
        if any(word in query_lower for word in ["compare", "correlation", "relationship"]):
            classifications.append("comparative_analysis")
        if any(word in query_lower for word in ["trend", "time", "temporal"]):
            classifications.append("temporal_analysis")
        if any(word in query_lower for word in ["map", "spatial", "geographic"]):
            classifications.append("spatial_analysis")
        if any(word in query_lower for word in ["profile", "vertical", "depth"]):
            classifications.append("vertical_analysis")
        
        # Complexity classification
        if len(classifications) > 2:
            classifications.append("complex_multi_parameter")
        elif len(classifications) == 0:
            classifications.append("general_exploration")
        
        return ", ".join(classifications)
    
    def _get_oceanographic_context(self, query: str) -> str:
        """Provide oceanographic context for the query"""
        query_lower = query.lower()
        
        context_notes = []
        
        if "temperature" in query_lower:
            context_notes.append("Temperature profiles show thermocline structure and mixed layer depth")
        if "salinity" in query_lower:
            context_notes.append("Salinity indicates water mass properties and circulation patterns")
        if "oxygen" in query_lower:
            context_notes.append("Oxygen levels indicate biological activity and water mass age")
        if "indian ocean" in query_lower:
            context_notes.append("Indian Ocean: Monsoon-driven circulation, warm pool dynamics")
        if "pacific" in query_lower:
            context_notes.append("Pacific Ocean: ENSO variability, equatorial upwelling, deep water formation")
        if "atlantic" in query_lower:
            context_notes.append("Atlantic Ocean: Meridional overturning circulation, Gulf Stream dynamics")
        
        return "; ".join(context_notes) if context_notes else "General oceanographic data analysis"
    
    def _determine_enhanced_visualization(self, query: str, results: List[Dict]) -> str:
        """Determine optimal visualization with enhanced intelligence"""
        query_lower = query.lower()
        
        # Check data characteristics
        has_depth = any('depth' in result for result in results if isinstance(result, dict))
        has_coords = any(all(k in result for k in ['latitude', 'longitude']) for result in results if isinstance(result, dict))
        has_time = any('timestamp' in result for result in results if isinstance(result, dict))
        
        # Enhanced visualization logic
        if any(word in query_lower for word in ["profile", "vertical", "depth"]) and has_depth:
            return "profile_plot"
        elif any(word in query_lower for word in ["map", "spatial", "geographic"]) and has_coords:
            return "map_plot"
        elif any(word in query_lower for word in ["time", "trend", "temporal"]) and has_time:
            return "time_series"
        elif any(word in query_lower for word in ["compare", "correlation"]):
            return "scatter_plot"
        elif "float" in query_lower and has_coords:
            return "float_trajectory"
        elif any(word in query_lower for word in ["bgc", "oxygen", "chlorophyll"]):
            return "bgc_plot"
        else:
            return "enhanced_dashboard"
    
    def _get_viz_recommendations(self, query: str) -> List[str]:
        """Get visualization recommendations"""
        recommendations = []
        query_lower = query.lower()
        
        if "temperature" in query_lower:
            recommendations.extend([
                "Temperature vs depth profile plots",
                "Temperature contour maps",
                "Temperature time series analysis"
            ])
        
        if "salinity" in query_lower:
            recommendations.extend([
                "T-S (Temperature-Salinity) diagrams",
                "Salinity section plots",
                "Halocline depth analysis"
            ])
        
        if any(word in query_lower for word in ["oxygen", "chlorophyll", "bgc"]):
            recommendations.extend([
                "BGC parameter vs depth profiles",
                "Oxygen minimum zone mapping",
                "Chlorophyll surface distribution"
            ])
        
        if "float" in query_lower:
            recommendations.extend([
                "Float trajectory mapping",
                "Float status dashboard",
                "Data collection timeline"
            ])
        
        return recommendations[:5]  # Limit to top 5
    
    def _generate_follow_up_suggestions(self, query: str) -> List[str]:
        """Generate intelligent follow-up research suggestions"""
        suggestions = []
        query_lower = query.lower()
        
        if "temperature" in query_lower:
            suggestions.extend([
                "Analyze seasonal temperature variations",
                "Compare with historical temperature data",
                "Investigate mixed layer depth changes"
            ])
        
        if "salinity" in query_lower:
            suggestions.extend([
                "Examine salinity-temperature relationships",
                "Study freshwater input effects",
                "Analyze density stratification"
            ])
        
        if any(word in query_lower for word in ["oxygen", "bgc"]):
            suggestions.extend([
                "Investigate oxygen minimum zones",
                "Analyze primary productivity patterns",
                "Study carbon cycle indicators"
            ])
        
        if "region" in query_lower or any(ocean in query_lower for ocean in ["indian", "pacific", "atlantic"]):
            suggestions.extend([
                "Compare with other ocean basins",
                "Analyze regional climate impacts",
                "Study water mass characteristics"
            ])
        
        return suggestions[:4]  # Limit to top 4
    
    def _get_data_quality_notes(self, query: str) -> str:
        """Provide data quality and uncertainty information"""
        notes = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["oxygen", "chlorophyll", "bgc"]):
            notes.append("BGC data has lower spatial coverage than core T/S measurements")
        
        if "recent" in query_lower or "latest" in query_lower:
            notes.append("Recent data may be preliminary and subject to delayed-mode quality control")
        
        if any(region in query_lower for region in ["arctic", "southern", "polar"]):
            notes.append("Polar regions have seasonal data gaps due to ice coverage")
        
        if "deep" in query_lower or "bottom" in query_lower:
            notes.append("Deep ocean measurements are less frequent and have higher uncertainty")
        
        base_note = "All ARGO data undergoes real-time and delayed-mode quality control procedures"
        
        if notes:
            return f"{base_note}. Additional notes: {'; '.join(notes)}"
        return base_note
    
    def _extract_scientific_insights(self, ai_response: str) -> List[str]:
        """Extract key scientific insights from AI response"""
        insights = []
        
        # Look for key oceanographic concepts
        response_lower = ai_response.lower()
        
        if "thermocline" in response_lower:
            insights.append("Thermocline structure analysis relevant")
        if "mixed layer" in response_lower:
            insights.append("Mixed layer dynamics important")
        if "upwelling" in response_lower:
            insights.append("Upwelling processes detected")
        if "water mass" in response_lower:
            insights.append("Water mass analysis applicable")
        if "circulation" in response_lower:
            insights.append("Ocean circulation patterns relevant")
        
        return insights
    
    def _extract_sql_from_response(self, response: str, query: str) -> str:
        """Extract SQL query from AI response or generate fallback"""
        # Look for SQL in the response
        sql_match = re.search(r'SELECT.*?;', response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            return sql_match.group(0)
        
        # Generate fallback SQL
        return self._generate_fallback_sql(query)
    
    def _generate_fallback_sql(self, query: str) -> str:
        """Generate fallback SQL query"""
        query_lower = query.lower()
        
        if "float" in query_lower:
            return "SELECT * FROM argo_floats WHERE status = 'active' LIMIT 50;"
        elif any(word in query_lower for word in ["oxygen", "chlorophyll", "bgc"]):
            return "SELECT profile_id, latitude, longitude, depth, oxygen, chlorophyll, nitrate, ph FROM argo_profiles WHERE oxygen IS NOT NULL OR chlorophyll IS NOT NULL LIMIT 100;"
        else:
            return "SELECT * FROM argo_profiles ORDER BY timestamp DESC LIMIT 100;"
    
    def _rule_based_analysis(self, query: str) -> str:
        """Rule-based analysis fallback when LLM is not available"""
        return f"""
        🔍 Query Analysis: {query}
        
        📊 Recommended Approach:
        - Execute targeted SQL query for data retrieval
        - Apply oceanographic domain knowledge
        - Generate appropriate visualizations
        
        🌊 Scientific Context:
        This query involves ARGO oceanographic data analysis with focus on understanding ocean patterns and processes.
        """
    
    def _create_error_response(self, query: str, error: str) -> QueryResponse:
        """Create error response with helpful information"""
        return QueryResponse(
            query=query,
            sql_query="-- Error in query processing",
            results=[],
            visualization_type="error",
            metadata={
                "error": error,
                "natural_response": f"I encountered an error processing your oceanographic query: {error}. Please try rephrasing your question or check the data parameters.",
                "suggestions": [
                    "Check query syntax and parameters",
                    "Verify data availability for requested region/time",
                    "Try a simpler query first",
                    "Contact system administrator if problem persists"
                ]
            }
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.mcp_enabled:
            await self.mcp_manager.cleanup()
        logger.info("Enhanced RAG pipeline cleaned up")

# Global enhanced pipeline instance
enhanced_rag_pipeline = EnhancedArgoRAGPipeline()

async def initialize_enhanced_rag():
    """Initialize the enhanced RAG pipeline"""
    return await enhanced_rag_pipeline.initialize()

async def process_query_enhanced(request: QueryRequest) -> QueryResponse:
    """Process query using enhanced RAG pipeline"""
    return await enhanced_rag_pipeline.process_enhanced_query(request)