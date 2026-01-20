"""
Enhanced ARGO RAG Pipeline with local LLM and MCP support
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# Import our new components
from db.adapter import db
from embeddings.service import vector_store
from llm.runner import llm_runner
from sql.templates import sql_engine, param_extractor
from models.data_models import QueryRequest, QueryResponse
from config import config

logger = logging.getLogger(__name__)

class EnhancedArgoRAGPipeline:
    """Enhanced RAG pipeline with local LLM, MCP, and comprehensive data access"""
    
    def __init__(self):
        self.db = db
        self.vector_store = vector_store
        self.llm_runner = llm_runner
        self.sql_engine = sql_engine
        self.param_extractor = param_extractor
        
        # Initialize metadata summarization
        self._initialize_metadata_cache()
        
        logger.info("Enhanced ARGO RAG Pipeline initialized")
    
    def _initialize_metadata_cache(self):
        """Initialize metadata summaries if not present"""
        try:
            # Check if we have metadata summaries
            summaries = self.db.execute_query(
                "SELECT COUNT(*) as count FROM metadata_summaries"
            )
            
            if summaries and summaries[0]['count'] == 0:
                logger.info("No metadata summaries found, generating...")
                self._generate_metadata_summaries()
            else:
                logger.info(f"Found {summaries[0]['count']} metadata summaries")
                
        except Exception as e:
            logger.error(f"Failed to initialize metadata cache: {e}")
    
    def _generate_metadata_summaries(self):
        """Generate metadata summaries for profiles"""
        try:
            # Get recent profiles
            profiles = self.db.get_profiles(limit=100)
            
            for profile in profiles:
                summary_text = self._create_profile_summary(profile)
                
                metadata = {
                    'profile_id': profile['profile_id'],
                    'float_id': profile['float_id'],
                    'datetime': profile['profile_datetime'],
                    'lat': profile['lat'],
                    'lon': profile['lon'],
                    'region': profile.get('region'),
                    'n_levels': profile.get('n_levels')
                }
                
                # Store in database
                summary_data = {
                    'type': 'profile',
                    'text_summary': summary_text,
                    'ref_id': profile['profile_id'],
                    'metadata_json': metadata
                }
                
                self.db.insert_metadata_summary(summary_data)
                
                # Add to vector store
                self.vector_store.add_profile_summary(
                    profile['profile_id'],
                    summary_text,
                    metadata
                )
            
            # Save vector store
            self.vector_store.save()
            logger.info(f"Generated metadata summaries for {len(profiles)} profiles")
            
        except Exception as e:
            logger.error(f"Failed to generate metadata summaries: {e}")
    
    def _create_profile_summary(self, profile: Dict[str, Any]) -> str:
        """Create natural language summary for a profile"""
        try:
            # Get measurements for this profile
            measurements = self.db.get_measurements(profile['profile_id'])
            
            # Build summary
            datetime_str = profile['profile_datetime']
            if isinstance(datetime_str, str):
                dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                date_str = dt.strftime('%Y-%m-%d %H:%M UTC')
            else:
                date_str = str(datetime_str)
            
            summary = f"Profile {profile['profile_id']} (float {profile['float_id']}) at {date_str} — "
            summary += f"lat {profile['lat']:.1f}, lon {profile['lon']:.1f}"
            
            if profile.get('region'):
                summary += f" in {profile['region']}"
            
            # Add measurement info
            if measurements:
                depths = [m['depth_m'] for m in measurements if m['depth_m'] is not None]
                temps = [m['temperature_c'] for m in measurements if m['temperature_c'] is not None]
                salinity = [m['salinity_psu'] for m in measurements if m['salinity_psu'] is not None]
                
                if depths:
                    summary += f"; depth range {min(depths):.0f}-{max(depths):.0f}m"
                
                if temps:
                    surface_temp = min(temps) if len(temps) > 1 else temps[0]
                    summary += f"; surface temp {surface_temp:.1f}°C"
                    
                    if len(temps) > 1:
                        deep_temp = temps[-1]
                        summary += f", deep temp {deep_temp:.1f}°C"
                
                if salinity:
                    sal_range = f"{min(salinity):.1f}-{max(salinity):.1f}"
                    summary += f"; salinity range {sal_range} PSU"
                
                # Check for BGC parameters
                bgc_params = []
                if any(m.get('oxygen_umol_kg') for m in measurements):
                    bgc_params.append('oxygen')
                if any(m.get('chlorophyll_mg_m3') for m in measurements):
                    bgc_params.append('chlorophyll')
                if any(m.get('ph_total') for m in measurements):
                    bgc_params.append('pH')
                if any(m.get('nitrate_umol_kg') for m in measurements):
                    bgc_params.append('nitrate')
                
                if bgc_params:
                    summary += f"; BGC: {', '.join(bgc_params)}"
            
            summary += "."
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create profile summary: {e}")
            return f"Profile {profile['profile_id']} at {profile['lat']:.1f}, {profile['lon']:.1f}"
    
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a natural language query using smart natural language understanding"""
        try:
            start_time = datetime.now()
            
            # Initialize smart processor if not already done
            if not hasattr(self, 'smart_processor'):
                from llm.smart_processor import SmartQueryProcessor
                self.smart_processor = SmartQueryProcessor(self.db)
            
            # Use smart processor for true natural language understanding
            smart_response = self.smart_processor.process_query(request.query)
            
            # Extract results and metadata
            sql_results = smart_response.get("results", [])
            sql_query = smart_response.get("sql_query", "")
            
            # Build final response
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response = QueryResponse(
                query=request.query,
                sql_query=sql_query,
                results=sql_results,
                result_count=len(sql_results),
                visualization_type=smart_response.get("visualization_type", "table"),
                metadata={
                    "intent": smart_response.get("intent", "general"),
                    "natural_response": smart_response.get("natural_response", ""),
                    "sources": [{"name": "smart_processor", "type": "intelligent_nlp"}],
                    "processing_time_ms": processing_time,
                    "llm_backend": "smart_processor",
                    "vector_backend": config.VECTOR_BACKEND,
                    "status": "success"
                }
            )
            
            logger.info(f"Smart query processed: {response.result_count} results")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return QueryResponse(
                query=request.query,
                sql_query="-- Query processing failed",
                results=[],
                result_count=0,
                visualization_type="error",
                metadata={
                    "error": str(e),
                    "status": "failed",
                    "processing_time_ms": processing_time,
                    "natural_response": f"I encountered an error processing your query: {str(e)}. Please try asking about ARGO ocean data in a different way."
                }
            )
    
    def _extract_query_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from natural language query"""
        params = {
            'limit': 100,  # Default limit
            'coordinates': self.param_extractor.extract_coordinates(query),
            'date_range': self.param_extractor.extract_date_range(query),
            'parameters': self.param_extractor.extract_parameters(query),
            'region': self.param_extractor.extract_region(query)
        }
        
        # Handle equatorial queries
        if 'equator' in query.lower() or 'equatorial' in query.lower():
            params['coordinates'] = {'lat': 0, 'lon': 0}
            params['lat_range'] = [-5, 5]
        
        return params
    
    def _retrieve_relevant_chunks(self, query: str, params: Dict[str, Any]) -> List[str]:
        """Retrieve relevant metadata chunks using vector search"""
        try:
            # Vector similarity search
            similar_results = self.vector_store.search_similar(
                query=query,
                top_k=config.TOP_K
            )
            
            chunks = []
            for result in similar_results:
                chunk_text = result.get('text_summary', '')
                if chunk_text:
                    chunks.append(chunk_text)
            
            # If no vector results, fall back to database query
            if not chunks:
                # Get recent profiles as fallback
                profiles = self.db.get_profiles(
                    filters=self._build_db_filters(params),
                    limit=5
                )
                
                for profile in profiles:
                    chunk_text = self._create_profile_summary(profile)
                    chunks.append(chunk_text)
            
            return chunks[:config.TOP_K]
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            return []
    
    def _build_db_filters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build database filters from extracted parameters"""
        filters = {}
        
        if params.get('region'):
            filters['region'] = params['region']
        
        if params.get('date_range'):
            date_range = params['date_range']
            filters['start_date'] = date_range['start_date']
            filters['end_date'] = date_range['end_date']
        
        if params.get('lat_range'):
            lat_range = params['lat_range']
            filters['lat_min'] = lat_range[0]
            filters['lat_max'] = lat_range[1]
        
        return filters
    
    def _execute_safe_sql(self, mcp_response: Dict[str, Any], query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute SQL query with safety validation"""
        try:
            sql_template = mcp_response.get('sql_template', '').strip()
            slots = mcp_response.get('slots', {})
            
            if not sql_template:
                logger.warning("No SQL template in MCP response")
                return []
            
            # Extract parameters from MCP response and query params
            sql_params = []
            
            # Count parameter placeholders
            param_count = sql_template.count('?')
            
            # Fill parameters based on query type
            if 'equator' in mcp_response.get('intent', '').lower():
                sql_params = [-5, 5, query_params.get('limit', 100)]
            elif 'nearest' in mcp_response.get('intent', '').lower():
                coords = query_params.get('coordinates', {'lat': 0, 'lon': 0})
                lat, lon = coords['lat'], coords['lon']
                sql_params = [lat, lat, lon, lon, lat, query_params.get('limit', 5)]
            elif 'region' in mcp_response.get('intent', '').lower():
                region = query_params.get('region', 'pacific')
                sql_params = [region, query_params.get('limit', 100)]
            else:
                # Default parameters
                sql_params = [query_params.get('limit', 100)]
            
            # Ensure we have the right number of parameters
            while len(sql_params) < param_count:
                sql_params.append(100)  # Default limit
            
            sql_params = sql_params[:param_count]  # Truncate if too many
            
            # Execute query
            results = self.db.execute_query(sql_template, tuple(sql_params))
            
            logger.info(f"Executed SQL query, got {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute SQL: {e}")
            logger.error(f"SQL: {mcp_response.get('sql_template')}")
            logger.error(f"Params: {query_params}")
            return []
    
    def _compose_response(self, request: QueryRequest, mcp_response: Dict[str, Any], 
                         sql_results: List[Dict[str, Any]], start_time: datetime) -> QueryResponse:
        """Compose final query response"""
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Determine visualization type
        viz_type = self._determine_visualization_type(request.query, sql_results)
        
        # Generate intelligent natural language response
        natural_response = self._generate_intelligent_summary(request.query, mcp_response, sql_results)
        
        # Build metadata
        metadata = {
            "intent": mcp_response.get('intent', 'Unknown'),
            "slots": mcp_response.get('slots', {}),
            "natural_response": natural_response,
            "sources": mcp_response.get('sources', []),
            "processing_time_ms": processing_time,
            "llm_backend": config.LLM_BACKEND,
            "vector_backend": config.VECTOR_BACKEND
        }
        
        return QueryResponse(
            query=request.query,
            sql_query=mcp_response.get('sql_template', ''),
            results=sql_results,
            result_count=len(sql_results),
            visualization_type=viz_type,
            metadata=metadata
        )
    
    def _determine_visualization_type(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Determine appropriate visualization type based on query and results"""
        query_lower = query.lower()
        
        if not results:
            return "table"
        
        # Check if results have geographic data
        has_coords = any('lat' in result and 'lon' in result for result in results)
        
        if has_coords and ('location' in query_lower or 'where' in query_lower or 'nearest' in query_lower):
            return "map"
        
        # Check for depth profile data
        has_depth = any('depth' in result or 'depth_m' in result for result in results)
        if has_depth and ('profile' in query_lower or 'depth' in query_lower):
            return "profile"
        
        # Check for comparison queries
        if 'compare' in query_lower or 'vs' in query_lower:
            return "comparison"
        
        # Check for time series
        has_time = any('datetime' in result or 'profile_datetime' in result for result in results)
        if has_time and ('trend' in query_lower or 'time' in query_lower):
            return "timeseries"
        
        # Default to table
        return "table"
    
    def _generate_intelligent_summary(self, query: str, mcp_response: Dict[str, Any], 
                                     results: List[Dict[str, Any]]) -> str:
        """Generate intelligent, conversational summary of query results"""
        if not results:
            return "No data found for your query. Please try refining your search criteria."
        
        query_lower = query.lower()
        intent = mcp_response.get('intent', '').lower()
        
        # Temperature-related summaries
        if 'temperature' in query_lower or 'temperature' in intent:
            return self._summarize_temperature_data(query, results)
        
        # Salinity-related summaries
        elif 'salinity' in query_lower or 'salinity' in intent:
            return self._summarize_salinity_data(query, results)
        
        # Float-related summaries
        elif 'float' in query_lower or 'float' in intent:
            return self._summarize_float_data(query, results)
        
        # Profile-related summaries
        elif 'profile' in query_lower or 'profile' in intent:
            return self._summarize_profile_data(query, results)
        
        # Location-related summaries
        elif any(word in query_lower for word in ['location', 'where', 'coordinates', 'lat', 'lon']):
            return self._summarize_location_data(query, results)
        
        # General data summary
        else:
            return self._summarize_general_data(query, results)
    
    def _summarize_temperature_data(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate temperature-specific summary"""
        query_lower = query.lower()
        
        # Handle highest/maximum temperature queries
        if any(word in query_lower for word in ['highest', 'maximum', 'max', 'peak']):
            if results and isinstance(results[0], dict):
                if any(word in query_lower for word in ['which', 'what', 'float']):
                    # Which float has highest temperature
                    float_id = results[0].get('float_id', 'Unknown')
                    max_temp = results[0].get('max_temperature', 0)
                    region = results[0].get('region', 'Unknown')
                    depth = results[0].get('depth_m', 0)
                    
                    return f"🌡️ **Highest Temperature Record**\n\n" \
                           f"• **Float ID**: {float_id}\n" \
                           f"• **Maximum Temperature**: {max_temp:.2f}°C\n" \
                           f"• **Region**: {region.replace('_', ' ').title()}\n" \
                           f"• **Depth**: {depth:.0f}m\n\n" \
                           f"This float recorded the highest temperature in the ARGO network dataset."
                else:
                    # Just highest temperature value
                    highest_temp = results[0].get('highest_temperature', 0)
                    float_id = results[0].get('float_id', 'Unknown')
                    depth = results[0].get('depth_m', 0)
                    
                    return f"🌡️ **Maximum Temperature Record**\n\n" \
                           f"• **Highest Temperature**: {highest_temp:.2f}°C\n" \
                           f"• **Recorded by Float**: {float_id}\n" \
                           f"• **Depth**: {depth:.0f}m\n\n" \
                           f"This represents the peak temperature measurement across all active ARGO floats."
            else:
                return "No temperature data found for maximum temperature query."
        
        # Handle average temperature queries
        elif 'average' in query_lower or 'mean' in query_lower:
            # Statistical summary
            if results and isinstance(results[0], dict):
                avg_temp = results[0].get('avg_temperature', 0)
                count = results[0].get('measurement_count', 0)
                min_temp = results[0].get('min_temp', 0)  # Fixed field name
                max_temp = results[0].get('max_temp', 0)  # Fixed field name
            else:
                avg_temp = count = min_temp = max_temp = 0
            
            return f"📊 **Temperature Analysis Summary**\n\n" \
                   f"• **Average Temperature**: {avg_temp:.2f}°C\n" \
                   f"• **Temperature Range**: {min_temp:.2f}°C to {max_temp:.2f}°C\n" \
                   f"• **Total Measurements**: {count:,} data points\n\n" \
                   f"The ARGO float network shows consistent oceanic temperature patterns across active monitoring stations."
        else:
            # Profile data summary
            temp_values = [r.get('temperature_c', 0) for r in results if r.get('temperature_c')]
            depths = [r.get('depth_m', 0) for r in results if r.get('depth_m')]
            
            if temp_values and depths:
                avg_temp = sum(temp_values) / len(temp_values)
                max_depth = max(depths) if depths else 0
                unique_profiles = len(set(r.get('profile_id', '') for r in results))
                
                return f"🌡️ **Temperature Profile Summary**\n\n" \
                       f"• **{len(results)} measurements** from **{unique_profiles} profiles**\n" \
                       f"• **Average Temperature**: {avg_temp:.2f}°C\n" \
                       f"• **Maximum Depth**: {max_depth:.0f}m\n" \
                       f"• **Temperature Range**: {min(temp_values):.2f}°C to {max(temp_values):.2f}°C\n\n" \
                       f"Data shows oceanic temperature variations across different depths and locations."
            else:
                return f"Found {len(results)} temperature measurements from active ARGO floats."
    
    def _summarize_salinity_data(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate salinity-specific summary"""
        if 'average' in query.lower() or 'mean' in query.lower():
            # Statistical summary
            if results and isinstance(results[0], dict):
                avg_sal = results[0].get('avg_salinity', 0)
                count = results[0].get('measurement_count', 0)
                min_sal = results[0].get('min_salinity', 0)  # Correct field name
                max_sal = results[0].get('max_salinity', 0)  # Correct field name
            else:
                avg_sal = count = min_sal = max_sal = 0
            
            return f"🧂 **Salinity Analysis Summary**\n\n" \
                   f"• **Average Salinity**: {avg_sal:.2f} PSU\n" \
                   f"• **Salinity Range**: {min_sal:.2f} to {max_sal:.2f} PSU\n" \
                   f"• **Total Measurements**: {count:,} data points\n\n" \
                   f"The salinity data indicates typical oceanic salt concentration levels across monitored regions."
        else:
            # Profile data summary
            sal_values = [r.get('salinity_psu', 0) for r in results if r.get('salinity_psu')]
            depths = [r.get('depth_m', 0) for r in results if r.get('depth_m')]
            
            if sal_values and depths:
                avg_sal = sum(sal_values) / len(sal_values)
                max_depth = max(depths) if depths else 0
                unique_profiles = len(set(r.get('profile_id', '') for r in results))
                
                return f"🌊 **Salinity Profile Summary**\n\n" \
                       f"• **{len(results)} measurements** from **{unique_profiles} profiles**\n" \
                       f"• **Average Salinity**: {avg_sal:.2f} PSU\n" \
                       f"• **Maximum Depth**: {max_depth:.0f}m\n" \
                       f"• **Salinity Range**: {min(sal_values):.2f} to {max(sal_values):.2f} PSU\n\n" \
                       f"Data reveals salinity variations across oceanic depths and geographic locations."
            else:
                return f"Found {len(results)} salinity measurements from active ARGO floats."
    
    def _summarize_float_data(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate float-specific summary"""
        if not results:
            return "No active ARGO floats found matching your criteria."
        
        query_lower = query.lower()
        
        # Handle list/name queries
        if any(word in query_lower for word in ['name', 'list', 'all', 'which']):
            if len(results) <= 10:  # Show detailed list for small numbers
                float_list = []
                for r in results:
                    float_id = r.get('float_id', 'Unknown')
                    region = r.get('region', 'Unknown').replace('_', ' ').title()
                    profiles = r.get('recent_profiles', 0)
                    float_list.append(f"• **{float_id}** - {region} ({profiles} profiles)")
                
                return f"🎯 **Active ARGO Float Directory**\n\n" + "\n".join(float_list) + \
                       f"\n\n**Total Active Floats**: {len(results)} operational units"
            else:
                # Summarize for large numbers
                regions = [r.get('region', 'Unknown') for r in results]
                unique_regions = list(set(regions))
                total_profiles = sum(r.get('recent_profiles', 0) for r in results)
                
                return f"🎯 **Active ARGO Float Directory**\n\n" \
                       f"• **Total Active Floats**: {len(results)} operational units\n" \
                       f"• **Coverage Regions**: {', '.join(unique_regions[:5])}{'...' if len(unique_regions) > 5 else ''}\n" \
                       f"• **Total Recent Profiles**: {total_profiles:,}\n\n" \
                       f"Too many floats to list individually. Use filters to narrow results."
        
        # Handle count queries
        elif any(word in query_lower for word in ['how many', 'count', 'number']):
            if results and isinstance(results[0], dict):
                count = results[0].get('active_count', len(results))
                regions = results[0].get('regions_covered', 0)
                total_profiles = results[0].get('total_profiles', 0)
                
                return f"📊 **ARGO Float Count Summary**\n\n" \
                       f"• **Active Floats**: {count} operational units\n" \
                       f"• **Regions Covered**: {regions} ocean regions\n" \
                       f"• **Total Profiles**: {total_profiles:,} data collections\n\n" \
                       f"The ARGO network maintains global ocean monitoring coverage."
            else:
                return f"**Active Float Count**: {len(results)} operational ARGO floats"
        
        # General float info
        else:
            # Count active floats
            float_ids = set(r.get('float_id', '') for r in results if r.get('float_id'))
            active_count = len(float_ids)
            
            # Geographic distribution
            regions = [r.get('region', 'Unknown') for r in results if r.get('region')]
            unique_regions = list(set(regions))
            
            return f"🎯 **ARGO Float Network Summary**\n\n" \
                   f"• **Active Floats**: {active_count} operational units\n" \
                   f"• **Coverage Regions**: {', '.join(unique_regions[:3])}{'...' if len(unique_regions) > 3 else ''}\n" \
                   f"• **Total Records**: {len(results)} float entries\n\n" \
                   f"The ARGO network maintains extensive global ocean monitoring coverage."
    
    def _summarize_profile_data(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate profile-specific summary"""
        if not results:
            return "No oceanographic profiles found for your query."
        
        # Profile statistics
        profile_ids = set(r.get('profile_id', '') for r in results if r.get('profile_id'))
        unique_profiles = len(profile_ids)
        
        # Recent activity
        dates = [r.get('profile_datetime', '') for r in results if r.get('profile_datetime')]
        if dates:
            recent_date = max(dates)
            return f"📈 **Oceanographic Profile Summary**\n\n" \
                   f"• **Profiles Found**: {unique_profiles} unique profiles\n" \
                   f"• **Total Data Points**: {len(results)} measurements\n" \
                   f"• **Most Recent**: {recent_date[:10] if recent_date else 'Unknown'}\n\n" \
                   f"Profiles contain comprehensive vertical ocean structure data from ARGO floats."
        else:
            return f"Found {unique_profiles} oceanographic profiles with {len(results)} total measurements."
    
    def _summarize_location_data(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate location-specific summary"""
        if not results:
            return "No data found for the specified location criteria."
        
        # Geographic bounds
        lats = [r.get('lat', 0) for r in results if r.get('lat')]
        lons = [r.get('lon', 0) for r in results if r.get('lon')]
        
        if lats and lons:
            lat_range = f"{min(lats):.2f}° to {max(lats):.2f}°"
            lon_range = f"{min(lons):.2f}° to {max(lons):.2f}°"
            
            return f"🗺️ **Geographic Data Summary**\n\n" \
                   f"• **Latitude Range**: {lat_range}\n" \
                   f"• **Longitude Range**: {lon_range}\n" \
                   f"• **Data Points**: {len(results)} measurements\n" \
                   f"• **Coverage Area**: Multi-point oceanic region\n\n" \
                   f"Data spans across diverse oceanic locations with comprehensive geographic coverage."
        else:
            return f"Found {len(results)} data points with location information."
    
    def _summarize_general_data(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate general data summary"""
        if not results:
            return "No data found matching your query criteria."
        
        # General statistics
        result_count = len(results)
        
        # Identify data types present
        data_types = []
        if any(r.get('temperature_c') for r in results):
            data_types.append("temperature")
        if any(r.get('salinity_psu') for r in results):
            data_types.append("salinity")
        if any(r.get('pressure_dbar') for r in results):
            data_types.append("pressure")
        
        return f"📊 **Ocean Data Summary**\n\n" \
               f"• **Records Found**: {result_count} measurements\n" \
               f"• **Data Types**: {', '.join(data_types) if data_types else 'Mixed parameters'}\n" \
               f"• **Source**: ARGO float network\n\n" \
               f"Comprehensive oceanographic dataset providing insights into marine conditions."
    
    def get_data_sample(self, sample_type: str = "recent", limit: int = 20) -> Dict[str, Any]:
        """Get sample data for testing and exploration"""
        try:
            if sample_type == "recent":
                profiles = self.db.get_profiles(limit=limit)
                return {
                    "type": "recent_profiles",
                    "data": profiles,
                    "count": len(profiles)
                }
            
            elif sample_type == "floats":
                floats = self.db.get_floats(limit=limit)
                return {
                    "type": "active_floats",
                    "data": floats,
                    "count": len(floats)
                }
            
            elif sample_type == "regions":
                # Get profile count by region
                results = self.db.execute_query("""
                    SELECT f.region, COUNT(p.profile_id) as profile_count,
                           COUNT(DISTINCT f.float_id) as float_count
                    FROM floats f
                    LEFT JOIN profiles p ON f.float_id = p.float_id
                    GROUP BY f.region
                    ORDER BY profile_count DESC
                """)
                return {
                    "type": "regional_summary",
                    "data": results,
                    "count": len(results)
                }
            
            else:
                return {"error": f"Unknown sample type: {sample_type}"}
                
        except Exception as e:
            logger.error(f"Failed to get data sample: {e}")
            return {"error": str(e)}
    
    def get_nearest_floats(self, lat: float, lon: float, date: datetime = None, limit: int = 5) -> Dict[str, Any]:
        """Get nearest ARGO floats to coordinates"""
        try:
            floats = self.db.get_nearest_floats(lat, lon, date, limit)
            
            return {
                "query_coordinates": {"lat": lat, "lon": lon},
                "query_date": date.isoformat() if date else None,
                "nearest_floats": floats,
                "count": len(floats)
            }
            
        except Exception as e:
            logger.error(f"Failed to get nearest floats: {e}")
            return {"error": str(e)}
    
    def get_analysis(self, analysis_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform specific analysis on ARGO data"""
        try:
            params = params or {}
            
            if analysis_type == "temperature_trends":
                # Analyze temperature trends over time
                results = self.db.execute_query("""
                    SELECT DATE(p.profile_datetime) as date,
                           AVG(m.temperature_c) as avg_temperature,
                           COUNT(m.meas_id) as measurement_count
                    FROM profiles p
                    JOIN measurements m ON p.profile_id = m.profile_id
                    WHERE m.temperature_c IS NOT NULL
                    AND p.profile_datetime >= ?
                    GROUP BY DATE(p.profile_datetime)
                    ORDER BY date DESC
                    LIMIT 30
                """, (datetime.now() - timedelta(days=30),))
                
                return {
                    "analysis_type": "temperature_trends",
                    "data": results,
                    "count": len(results)
                }
            
            elif analysis_type == "salinity_distribution":
                # Analyze salinity distribution by region
                results = self.db.execute_query("""
                    SELECT f.region,
                           AVG(m.salinity_psu) as avg_salinity,
                           MIN(m.salinity_psu) as min_salinity,
                           MAX(m.salinity_psu) as max_salinity,
                           COUNT(m.meas_id) as measurement_count
                    FROM profiles p
                    JOIN measurements m ON p.profile_id = m.profile_id
                    JOIN floats f ON p.float_id = f.float_id
                    WHERE m.salinity_psu IS NOT NULL
                    GROUP BY f.region
                    ORDER BY avg_salinity DESC
                """)
                
                return {
                    "analysis_type": "salinity_distribution",
                    "data": results,
                    "count": len(results)
                }
            
            elif analysis_type == "depth_profiles":
                # Get depth profile statistics
                results = self.db.execute_query("""
                    SELECT CAST(m.depth_m/100 AS INTEGER)*100 as depth_bin,
                           AVG(m.temperature_c) as avg_temperature,
                           AVG(m.salinity_psu) as avg_salinity,
                           COUNT(m.meas_id) as measurement_count
                    FROM measurements m
                    WHERE m.temperature_c IS NOT NULL AND m.salinity_psu IS NOT NULL
                    GROUP BY depth_bin
                    ORDER BY depth_bin
                    LIMIT 20
                """)
                
                return {
                    "analysis_type": "depth_profiles",
                    "data": results,
                    "count": len(results)
                }
            
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}
    
    def export_data(self, query: str, format: str = "json", limit: int = 1000) -> Dict[str, Any]:
        """Export query results in specified format"""
        try:
            # Process query to get results
            request = QueryRequest(query=query, limit=limit)
            response = self.process_query(request)
            
            if format.lower() == "json":
                return {
                    "format": "json",
                    "query": query,
                    "data": response.results,
                    "metadata": response.metadata
                }
            
            elif format.lower() == "csv":
                import io
                import csv
                
                if not response.results:
                    return {"error": "No data to export"}
                
                output = io.StringIO()
                if response.results:
                    fieldnames = response.results[0].keys()
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(response.results)
                
                return {
                    "format": "csv",
                    "query": query,
                    "data": output.getvalue(),
                    "metadata": response.metadata
                }
            
            else:
                return {"error": f"Unsupported export format: {format}"}
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {"error": str(e)}
    
    def get_meta_info(self) -> Dict[str, Any]:
        """Get dataset metadata and system information"""
        try:
            dataset_meta = self.db.get_dataset_metadata()
            
            # Add system information
            dataset_meta.update({
                "system": {
                    "llm_backend": config.LLM_BACKEND,
                    "vector_backend": config.VECTOR_BACKEND,
                    "embedding_model": config.EMBEDDING_MODEL,
                    "database_type": "SQLite" if config.USE_SQLITE else "PostgreSQL"
                },
                "capabilities": {
                    "natural_language_query": True,
                    "vector_search": self.vector_store.backend is not None,
                    "sql_generation": True,
                    "export_formats": ["json", "csv"],
                    "visualization_types": ["map", "profile", "timeseries", "comparison", "table"]
                }
            })
            
            return dataset_meta
            
        except Exception as e:
            logger.error(f"Failed to get meta info: {e}")
            return {"error": str(e)}

# Global enhanced pipeline instance
enhanced_rag_pipeline = EnhancedArgoRAGPipeline()