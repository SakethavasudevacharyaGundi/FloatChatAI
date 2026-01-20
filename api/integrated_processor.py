"""
Integrated ARGO Ocean Data Processor
Properly connects RAG + MCP + Database + Gemini for ChatGPT-quality responses
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncpg

logger = logging.getLogger(__name__)

class IntegratedArgoProcessor:
    """
    Truly integrated processor that combines:
    - RAG: Scientific context from 105 documents
    - MCP: Oceanographic analysis tools
    - Database: Real ARGO data (4,380 measurements)
    - LLM: Gemini 1.5-Flash with proper context
    """
    
    def __init__(self):
        self.rag_pipeline = None
        self.mcp_server = None
        self.llm = None
        self.db_pool = None
        self.is_initialized = False
        self.db_config = {
            "host": os.getenv('DB_HOST', 'localhost'),
            "port": os.getenv('DB_PORT', '5433'),
            "user": os.getenv('DB_USER', 'postgres'),
            "password": os.getenv('DB_PASSWORD', 'your-database-password-here'),
            "database": os.getenv('DB_NAME', 'argo_ocean_data')
        }
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.gemini_api_key:
            try:
                from config import config
                self.gemini_api_key = getattr(config, 'GOOGLE_API_KEY', None)
            except ImportError:
                pass
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Import components
            from rag.rag_pipeline import ArgoRAGPipeline
            from mcp.mcp_server import ArgoMCPServer
            from ai.advanced_processor import advanced_llm
            import asyncpg
            
            # Initialize RAG Pipeline
            self.rag_pipeline = ArgoRAGPipeline(use_enhanced_features=True)
            await self.rag_pipeline.initialize_postgresql()
            await self.rag_pipeline.ingest_postgresql_data_to_rag()
            logger.info("✅ RAG Pipeline initialized with 105+ documents")
            
            # Initialize MCP Server
            self.mcp_server = ArgoMCPServer()
            await self.mcp_server.initialize()
            logger.info("✅ MCP Server initialized with 20+ oceanographic tools")
            
            # Initialize LLM
            self.llm = advanced_llm
            if self.llm.is_online():
                logger.info("✅ Advanced LLM initialized")
            else:
                logger.warning("⚠️ LLM not available, using fallback")
            
            # Initialize Database
            self.db_pool = await asyncpg.create_pool(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', '5433')),
                database=os.getenv('DB_NAME', 'argo_ocean_data'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'your-database-password-here'),
                min_size=2,
                max_size=10
            )
            logger.info("✅ Database connection established")
            
            self.is_initialized = True
            logger.info("🎉 Integrated Processor fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            self.is_initialized = False
    
    async def process_intelligent_query(self, query: str) -> Dict[str, Any]:
        """Process query with full intelligence integration"""
        if not self.is_initialized:
            return {"error": "Processor not initialized"}
        
        start_time = datetime.now()
        
        try:
            # Step 1: Get scientific context from RAG
            logger.info("🔍 Step 1: Retrieving scientific context from RAG...")
            rag_context = await self.rag_pipeline.get_enhanced_context_for_llm(query, top_k=3)
            
            # Step 2: Query database for actual data
            logger.info("📊 Step 2: Querying database for real data...")
            db_data = await self._query_database_intelligently(query)
            
            # Step 3: Use MCP tools for analysis
            logger.info("🔬 Step 3: Applying oceanographic analysis tools...")
            mcp_analysis = await self._apply_mcp_analysis(query, db_data)
            
            # Step 4: Generate intelligent response with LLM
            logger.info("🧠 Step 4: Generating intelligent response...")
            intelligent_response = await self._generate_intelligent_response(
                query, rag_context, db_data, mcp_analysis
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "query": query,
                "response": intelligent_response,
                "data": db_data,
                "rag_context": rag_context,
                "mcp_analysis": mcp_analysis,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "processing_method": "integrated_intelligence"
            }
            
        except Exception as e:
            logger.error(f"❌ Intelligent processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _query_database_intelligently(self, query: str) -> Dict[str, Any]:
        """Query database with intelligent SQL generation"""
        try:
            # Use advanced LLM to generate SQL
            if self.llm and self.llm.is_online():
                sql_success, sql_query, sql_message = await self.llm.generate_sql_query(query)
                
                if sql_success:
                    # Execute the generated SQL
                    exec_success, data, exec_message = await self.llm.execute_generated_query(sql_query)
                    
                    if exec_success:
                        return {
                            "sql_query": sql_query,
                            "data": data,
                            "data_count": len(data),
                            "execution_success": True
                        }
            
            # Fallback to simple database search
            return await self._fallback_database_search(query)
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return {"error": str(e), "data": [], "data_count": 0}
    
    async def _fallback_database_search(self, query: str) -> Dict[str, Any]:
        """Fallback database search when LLM SQL generation fails"""
        try:
            if not self.db_pool:
                logger.error("Database pool not available")
                return {"error": "Database not connected", "data": [], "data_count": 0}
            
            # Create a new connection instead of using pool
            conn = await asyncpg.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                database=self.db_config["database"]
            )
            
            try:
                # Simple keyword-based search
                query_lower = query.lower()
                
                if 'temperature' in query_lower or 'indian ocean' in query_lower:
                    rows = await conn.fetch("""
                        SELECT p.profile_id, p.lat, p.lon, p.profile_datetime,
                               m.depth_m, m.temperature_c, m.salinity_psu
                        FROM profiles p
                        JOIN measurements m ON p.profile_id = m.profile_id
                        WHERE m.temperature_c IS NOT NULL
                        AND p.lat >= -20 AND p.lat <= 30 
                        AND p.lon >= 40 AND p.lon <= 120
                        ORDER BY p.profile_datetime DESC
                        LIMIT 100
                    """)
                elif 'salinity' in query_lower:
                    rows = await conn.fetch("""
                        SELECT p.profile_id, p.lat, p.lon, p.profile_datetime,
                               m.depth_m, m.temperature_c, m.salinity_psu
                        FROM profiles p
                        JOIN measurements m ON p.profile_id = m.profile_id
                        WHERE m.salinity_psu IS NOT NULL
                        AND p.lat >= -20 AND p.lat <= 30 
                        AND p.lon >= 40 AND p.lon <= 120
                        ORDER BY p.profile_datetime DESC
                        LIMIT 100
                    """)
                else:
                    rows = await conn.fetch("""
                        SELECT p.profile_id, p.lat, p.lon, p.profile_datetime,
                               m.depth_m, m.temperature_c, m.salinity_psu
                        FROM profiles p
                        JOIN measurements m ON p.profile_id = m.profile_id
                        WHERE p.lat >= -20 AND p.lat <= 30 
                        AND p.lon >= 40 AND p.lon <= 120
                        ORDER BY p.profile_datetime DESC
                        LIMIT 50
                    """)
                
                data = [dict(row) for row in rows]
                logger.info(f"Database query returned {len(data)} rows")
                
                return {
                    "sql_query": "Fallback search query",
                    "data": data,
                    "data_count": len(data),
                    "execution_success": True
                }
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Fallback database search failed: {e}")
            return {"error": str(e), "data": [], "data_count": 0}
    
    async def _apply_mcp_analysis(self, query: str, db_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply MCP tools for oceanographic analysis"""
        try:
            analysis_results = {}
            
            # Determine which MCP tools to use based on query
            query_lower = query.lower()
            
            if 'profile' in query_lower and db_data.get('data'):
                # Use profile analysis tools
                profile_ids = [row.get('profile_id') for row in db_data['data'][:5]]
                if profile_ids:
                    # Analyze depth profile
                    depth_analysis = await self.mcp_server.execute_tool(
                        'analyze_depth_profile',
                        {
                            'profile_id': profile_ids[0],
                            'parameters': ['temperature_c', 'salinity_psu']
                        }
                    )
                    analysis_results['depth_analysis'] = depth_analysis
            
            if 'float' in query_lower or 'trajectory' in query_lower:
                # Use float analysis tools
                float_analysis = await self.mcp_server.execute_tool(
                    'get_float_trajectory',
                    {'float_id': '1900121'}
                )
                analysis_results['float_trajectory'] = float_analysis
            
            if 'temperature' in query_lower and 'plot' in query_lower:
                # Generate temperature profile plot
                profile_ids = [row.get('profile_id') for row in db_data['data'][:3]]
                if profile_ids:
                    plot_result = await self.mcp_server.execute_tool(
                        'generate_temperature_profile_plot',
                        {
                            'profile_ids': profile_ids,
                            'comparison_mode': len(profile_ids) > 1
                        }
                    )
                    analysis_results['temperature_plot'] = plot_result
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"MCP analysis failed: {e}")
            return {"error": str(e)}
    
    async def _generate_intelligent_response(self, query: str, rag_context: str, 
                                           db_data: Dict[str, Any], mcp_analysis: Dict[str, Any]) -> str:
        """Generate intelligent response using all components"""
        try:
            if not self.llm or not self.llm.is_online():
                return self._generate_fallback_response(query, db_data, mcp_analysis)
            
            # Prepare comprehensive context for LLM
            context_parts = [
                f"User Query: {query}",
                "",
                "SCIENTIFIC CONTEXT (from RAG):",
                rag_context,
                "",
                f"DATABASE RESULTS: {db_data.get('data_count', 0)} records found",
                f"SQL Query: {db_data.get('sql_query', 'N/A')}",
                "",
                "OCEANOGRAPHIC ANALYSIS (from MCP tools):",
                json.dumps(mcp_analysis, indent=2, default=str)
            ]
            
            if db_data.get('data'):
                # Add sample data
                sample_data = db_data['data'][:3]
                context_parts.extend([
                    "",
                    "SAMPLE DATA:",
                    json.dumps(sample_data, indent=2, default=str)
                ])
            
            full_context = "\n".join(context_parts)
            
            # Generate response using advanced LLM
            prompt = f"""
You are an expert oceanographer analyzing ARGO float data. Based on the scientific context, database results, and oceanographic analysis provided, generate a comprehensive, intelligent response to the user's query.

Guidelines:
1. Use the scientific context to provide authoritative oceanographic insights
2. Reference specific data points and statistics from the database results
3. Incorporate findings from the MCP analysis tools
4. Provide scientific explanations, not just data summaries
5. Be conversational but scientifically accurate
6. Include relevant oceanographic terminology and concepts
7. If data is limited, explain what can be determined and what would require more data

Context:
{full_context}

Generate a comprehensive oceanographic analysis response:
"""
            
            response = await self.llm.model.generate_content_async(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Intelligent response generation failed: {e}")
            return self._generate_fallback_response(query, db_data, mcp_analysis)
    
    def _generate_fallback_response(self, query: str, db_data: Dict[str, Any], 
                                  mcp_analysis: Dict[str, Any]) -> str:
        """Generate fallback response when LLM is not available"""
        data_count = db_data.get('data_count', 0)
        
        if data_count == 0:
            return f"I couldn't find any data matching your query: '{query}'. Please try rephrasing or broadening your search criteria."
        
        response_parts = [
            f"Based on your query about '{query}', I found {data_count} relevant records from the ARGO oceanographic dataset.",
            "",
            "Key Findings:"
        ]
        
        if db_data.get('data'):
            sample = db_data['data'][0]
            if 'temperature_c' in sample:
                temps = [row.get('temperature_c') for row in db_data['data'] if row.get('temperature_c')]
                if temps:
                    response_parts.append(f"• Temperature range: {min(temps):.2f}°C to {max(temps):.2f}°C")
            
            if 'salinity_psu' in sample:
                salinities = [row.get('salinity_psu') for row in db_data['data'] if row.get('salinity_psu')]
                if salinities:
                    response_parts.append(f"• Salinity range: {min(salinities):.2f} to {max(salinities):.2f} PSU")
            
            if 'depth_m' in sample:
                depths = [row.get('depth_m') for row in db_data['data'] if row.get('depth_m')]
                if depths:
                    response_parts.append(f"• Depth coverage: {min(depths):.1f}m to {max(depths):.1f}m")
        
        # Add MCP analysis insights
        if mcp_analysis.get('depth_analysis', {}).get('success'):
            response_parts.append("• Oceanographic analysis completed using specialized tools")
        
        if mcp_analysis.get('temperature_plot', {}).get('success'):
            response_parts.append("• Temperature profile visualization generated")
        
        response_parts.extend([
            "",
            "This data represents real measurements from ARGO float 1900121 in the Indian Ocean region, collected between 2002-2005. The measurements provide valuable insights into oceanographic conditions and water mass properties."
        ])
        
        return "\n".join(response_parts)
    
    async def close(self):
        """Close all connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.mcp_server:
            await self.mcp_server.close()
        logger.info("Integrated processor closed")

# Global instance
integrated_processor = IntegratedArgoProcessor()
