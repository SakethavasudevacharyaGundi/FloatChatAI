"""
Enhanced RAG pipeline with Gemini AI for intelligent oceanographic data processing
Combines retrieval with Gemini-powered generation for natural language responses
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from vector_db.vector_store import VectorStore
from llm.gemini_processor import GeminiQueryProcessor
from config import Config
import json

logger = logging.getLogger(__name__)

class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with Gemini AI for oceanographic data"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.gemini_processor = GeminiQueryProcessor()
        self.context_limit = Config.CONTEXT_LIMIT
        
        logger.info("🧠 Enhanced RAG Pipeline with Gemini AI initialized")
    
    def query(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process query through enhanced RAG pipeline with Gemini AI"""
        
        try:
            logger.info(f"🔍 Processing query: {query[:100]}...")
            
            # Step 1: Analyze query intent using Gemini
            query_analysis = self._analyze_query_intent(query)
            
            # Step 2: Retrieve relevant data with enhanced search
            retrieved_data = self._enhanced_retrieval(query, query_analysis, filters)
            
            # Step 3: Process data for context
            processed_context = self._process_data_for_context(retrieved_data, query_analysis)
            
            # Step 4: Generate intelligent response using Gemini
            response = self._generate_intelligent_response(query, processed_context, query_analysis)
            
            # Step 5: Extract visualization suggestions
            viz_suggestions = self._extract_visualization_suggestions(query_analysis, retrieved_data)
            
            result = {
                'query': query,
                'response': response.get('response', 'Unable to generate response'),
                'data': retrieved_data,
                'processed_context': processed_context,
                'query_analysis': query_analysis,
                'visualization_suggestions': viz_suggestions,
                'metadata': {
                    'retrieved_count': len(retrieved_data),
                    'filters_applied': filters or {},
                    'intent_confidence': query_analysis.get('confidence', 0),
                    'gemini_response_type': response.get('type', 'unknown'),
                    'processing_time': response.get('processing_time', 0)
                }
            }
            
            logger.info(f"✅ RAG query processed successfully - {len(retrieved_data)} records retrieved")
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced RAG pipeline error: {e}")
            return self._create_error_response(query, str(e), filters)
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent using Gemini"""
        
        try:
            # Use Gemini to understand the query intent
            gemini_response = self.gemini_processor.process_query(
                f"Analyze this oceanographic query for intent, parameters, and requirements: '{query}'"
            )
            
            # Extract structured intent information
            if gemini_response.get('structured_data'):
                return gemini_response['structured_data']
            
            # Fallback to basic intent analysis
            return self._basic_intent_analysis(query)
            
        except Exception as e:
            logger.warning(f"⚠️ Gemini intent analysis failed, using fallback: {e}")
            return self._basic_intent_analysis(query)
    
    def _basic_intent_analysis(self, query: str) -> Dict[str, Any]:
        """Basic intent analysis fallback"""
        
        query_lower = query.lower()
        
        # Parameter detection
        parameters = []
        if any(word in query_lower for word in ['temperature', 'temp', 'thermal']):
            parameters.append('temperature')
        if any(word in query_lower for word in ['salinity', 'salt', 'psu']):
            parameters.append('salinity')
        if any(word in query_lower for word in ['oxygen', 'o2', 'dissolved']):
            parameters.append('oxygen')
        if any(word in query_lower for word in ['depth', 'pressure', 'deep']):
            parameters.append('depth')
        if any(word in query_lower for word in ['chlorophyll', 'chla', 'phyto']):
            parameters.append('chlorophyll')
        
        # Intent classification
        intent = 'general'
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            intent = 'comparison'
        elif any(word in query_lower for word in ['trend', 'time', 'temporal', 'over time']):
            intent = 'temporal_analysis'
        elif any(word in query_lower for word in ['map', 'location', 'geographic', 'spatial']):
            intent = 'spatial_analysis'
        elif any(word in query_lower for word in ['profile', 'vertical', 'depth']):
            intent = 'profile_analysis'
        elif any(word in query_lower for word in ['average', 'mean', 'statistics', 'summary']):
            intent = 'statistical_analysis'
        
        # Visualization hints
        viz_type = 'default'
        if 'profile' in query_lower or 'depth' in query_lower:
            viz_type = 'depth_profile'
        elif 'map' in query_lower or 'trajectory' in query_lower:
            viz_type = 'trajectory_map'
        elif 'time' in query_lower or 'trend' in query_lower:
            viz_type = 'time_series'
        elif 'compare' in query_lower or 'correlation' in query_lower:
            viz_type = 'scatter_plot'
        
        return {
            'intent': intent,
            'parameters': parameters,
            'visualization_type': viz_type,
            'confidence': 0.7,
            'geographic_focus': self._extract_geographic_focus(query),
            'temporal_focus': self._extract_temporal_focus(query)
        }
    
    def _extract_geographic_focus(self, query: str) -> Optional[str]:
        """Extract geographic focus from query"""
        
        query_lower = query.lower()
        
        regions = {
            'indian ocean': ['indian ocean', 'indian', 'arabian sea', 'bay of bengal'],
            'pacific': ['pacific', 'pacific ocean'],
            'atlantic': ['atlantic', 'atlantic ocean'],
            'southern ocean': ['southern ocean', 'antarctic'],
            'arctic': ['arctic', 'arctic ocean']
        }
        
        for region, keywords in regions.items():
            if any(keyword in query_lower for keyword in keywords):
                return region
        
        return None
    
    def _extract_temporal_focus(self, query: str) -> Optional[str]:
        """Extract temporal focus from query"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['recent', 'latest', 'current', 'now']):
            return 'recent'
        elif any(word in query_lower for word in ['historical', 'past', 'archive']):
            return 'historical'
        elif any(word in query_lower for word in ['seasonal', 'monthly', 'annual']):
            return 'seasonal'
        
        return None
    
    def _enhanced_retrieval(self, query: str, query_analysis: Dict[str, Any], 
                          filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced data retrieval based on query analysis"""
        
        # Enhance filters based on query analysis
        enhanced_filters = filters.copy() if filters else {}
        
        # Add parameter-specific filters
        if query_analysis.get('parameters'):
            # Prioritize records with requested parameters
            pass  # Implementation depends on vector store capabilities
        
        # Add geographic filters
        if query_analysis.get('geographic_focus'):
            enhanced_filters['region'] = query_analysis['geographic_focus']
        
        # Add temporal filters
        if query_analysis.get('temporal_focus') == 'recent':
            # Filter for recent data (implementation depends on data structure)
            pass
        
        # Increase retrieval limit for complex queries
        retrieval_limit = self.context_limit
        if query_analysis.get('intent') in ['comparison', 'statistical_analysis']:
            retrieval_limit = min(self.context_limit * 2, 1000)
        
        # Perform retrieval
        retrieved_data = self.vector_store.search(
            query, 
            limit=retrieval_limit, 
            filters=enhanced_filters
        )
        
        # Post-process retrieved data based on intent
        if query_analysis.get('intent') == 'comparison':
            retrieved_data = self._ensure_comparison_data(retrieved_data, query_analysis)
        
        return retrieved_data
    
    def _ensure_comparison_data(self, data: List[Dict[str, Any]], 
                              query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ensure retrieved data supports comparison analysis"""
        
        # Group by relevant dimensions for comparison
        if len(data) < 2:
            return data
        
        # Try to get diverse data for meaningful comparison
        # Implementation would depend on specific comparison needs
        return data
    
    def _process_data_for_context(self, data: List[Dict[str, Any]], 
                                query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process retrieved data to create rich context for Gemini"""
        
        if not data:
            return {'summary': 'No data available', 'statistics': {}}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data)
        
        # Calculate basic statistics
        statistics = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['temperature_c', 'salinity_psu', 'depth_m', 'oxygen_umol_kg', 'chlorophyll_mg_m3']:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    statistics[col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'count': int(len(col_data))
                    }
        
        # Geographic summary
        geographic_summary = {}
        if 'lat' in df.columns and 'lon' in df.columns:
            lat_data = df['lat'].dropna()
            lon_data = df['lon'].dropna()
            if len(lat_data) > 0:
                geographic_summary = {
                    'lat_range': [float(lat_data.min()), float(lat_data.max())],
                    'lon_range': [float(lon_data.min()), float(lon_data.max())],
                    'center': [float(lat_data.mean()), float(lon_data.mean())]
                }
        
        # Temporal summary
        temporal_summary = {}
        if 'profile_datetime' in df.columns:
            time_data = pd.to_datetime(df['profile_datetime'], errors='coerce').dropna()
            if len(time_data) > 0:
                temporal_summary = {
                    'date_range': [time_data.min().isoformat(), time_data.max().isoformat()],
                    'total_profiles': len(time_data),
                    'unique_floats': df['float_id'].nunique() if 'float_id' in df else 0
                }
        
        # Parameter-specific insights
        insights = []
        
        # Temperature insights
        if 'temperature_c' in statistics:
            temp_stats = statistics['temperature_c']
            if temp_stats['mean'] > 25:
                insights.append("Warm water conditions observed")
            elif temp_stats['mean'] < 10:
                insights.append("Cold water conditions observed")
            
            if temp_stats['std'] > 5:
                insights.append("High temperature variability")
        
        # Salinity insights
        if 'salinity_psu' in statistics:
            sal_stats = statistics['salinity_psu']
            if sal_stats['mean'] > 36:
                insights.append("High salinity waters")
            elif sal_stats['mean'] < 34:
                insights.append("Low salinity waters")
        
        # Depth insights
        if 'depth_m' in statistics:
            depth_stats = statistics['depth_m']
            if depth_stats['max'] > 1000:
                insights.append("Deep ocean measurements available")
            elif depth_stats['max'] < 200:
                insights.append("Surface layer focus")
        
        return {
            'data_summary': {
                'total_records': len(data),
                'available_parameters': list(statistics.keys()),
                'geographic_coverage': geographic_summary,
                'temporal_coverage': temporal_summary
            },
            'statistics': statistics,
            'insights': insights,
            'processed_for_intent': query_analysis.get('intent', 'general')
        }
    
    def _generate_intelligent_response(self, query: str, context: Dict[str, Any], 
                                     query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent response using Gemini with rich context"""
        
        try:
            # Build context-rich prompt for Gemini
            prompt = self._build_context_prompt(query, context, query_analysis)
            
            # Get Gemini response
            gemini_response = self.gemini_processor.process_query(prompt)
            
            return {
                'response': gemini_response.get('response', 'Unable to generate response'),
                'type': 'gemini_enhanced',
                'processing_time': gemini_response.get('processing_time', 0),
                'confidence': gemini_response.get('confidence', 0.8)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Gemini response generation failed, using fallback: {e}")
            return self._generate_fallback_response(query, context, query_analysis)
    
    def _build_context_prompt(self, query: str, context: Dict[str, Any], 
                            query_analysis: Dict[str, Any]) -> str:
        """Build context-rich prompt for Gemini"""
        
        prompt_parts = [
            "You are an expert oceanographer analyzing ARGO float data. ",
            f"User query: '{query}'\n",
            f"Query intent: {query_analysis.get('intent', 'general')}\n",
            f"Requested parameters: {', '.join(query_analysis.get('parameters', []))}\n\n"
        ]
        
        # Add data summary
        if context.get('data_summary'):
            summary = context['data_summary']
            prompt_parts.append(f"Data Overview:\n")
            prompt_parts.append(f"- {summary.get('total_records', 0)} oceanographic measurements\n")
            prompt_parts.append(f"- Parameters available: {', '.join(summary.get('available_parameters', []))}\n")
            
            if summary.get('geographic_coverage'):
                geo = summary['geographic_coverage']
                prompt_parts.append(f"- Geographic range: {geo.get('lat_range', [])}, {geo.get('lon_range', [])}\n")
            
            if summary.get('temporal_coverage'):
                temp = summary['temporal_coverage']
                prompt_parts.append(f"- Time range: {temp.get('date_range', [])}\n")
                prompt_parts.append(f"- {temp.get('total_profiles', 0)} profiles from {temp.get('unique_floats', 0)} floats\n")
        
        # Add statistical insights
        if context.get('statistics'):
            prompt_parts.append(f"\nKey Statistics:\n")
            for param, stats in context['statistics'].items():
                param_name = param.replace('_', ' ').title()
                prompt_parts.append(f"- {param_name}: {stats['mean']:.2f} ±{stats['std']:.2f} "
                                  f"(range: {stats['min']:.2f} to {stats['max']:.2f})\n")
        
        # Add insights
        if context.get('insights'):
            prompt_parts.append(f"\nKey Insights: {', '.join(context['insights'])}\n")
        
        prompt_parts.append(f"\nPlease provide a comprehensive, scientific response addressing the user's query. ")
        prompt_parts.append(f"Include specific data values, trends, and oceanographic interpretation where relevant. ")
        prompt_parts.append(f"Format the response to be informative yet accessible.")
        
        return ''.join(prompt_parts)
    
    def _generate_fallback_response(self, query: str, context: Dict[str, Any], 
                                  query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback response when Gemini is unavailable"""
        
        intent = query_analysis.get('intent', 'general')
        parameters = query_analysis.get('parameters', [])
        
        if intent == 'statistical_analysis':
            response = self._generate_statistical_response(context)
        elif intent == 'comparison':
            response = self._generate_comparison_response(context, parameters)
        elif intent == 'temporal_analysis':
            response = self._generate_temporal_response(context)
        elif intent == 'spatial_analysis':
            response = self._generate_spatial_response(context)
        else:
            response = self._generate_general_fallback_response(context)
        
        return {
            'response': response,
            'type': 'fallback',
            'processing_time': 0,
            'confidence': 0.6
        }
    
    def _generate_statistical_response(self, context: Dict[str, Any]) -> str:
        """Generate statistical analysis response"""
        
        statistics = context.get('statistics', {})
        if not statistics:
            return "No statistical data available for analysis."
        
        response_parts = ["Statistical Analysis of Oceanographic Data:\n\n"]
        
        for param, stats in statistics.items():
            param_name = param.replace('_', ' ').title()
            response_parts.append(f"**{param_name}:**\n")
            response_parts.append(f"- Mean: {stats['mean']:.2f}\n")
            response_parts.append(f"- Standard Deviation: {stats['std']:.2f}\n")
            response_parts.append(f"- Range: {stats['min']:.2f} to {stats['max']:.2f}\n")
            response_parts.append(f"- Sample Size: {stats['count']} measurements\n\n")
        
        if context.get('insights'):
            response_parts.append("Key Insights:\n")
            for insight in context['insights']:
                response_parts.append(f"- {insight}\n")
        
        return ''.join(response_parts)
    
    def _generate_comparison_response(self, context: Dict[str, Any], parameters: List[str]) -> str:
        """Generate comparison analysis response"""
        
        statistics = context.get('statistics', {})
        if len(statistics) < 2:
            return "Insufficient data for comparison analysis."
        
        response_parts = ["Comparative Analysis:\n\n"]
        
        # Compare parameters if multiple available
        param_names = list(statistics.keys())
        for i, param1 in enumerate(param_names):
            for param2 in param_names[i+1:]:
                response_parts.append(f"**{param1.replace('_', ' ').title()} vs {param2.replace('_', ' ').title()}:**\n")
                # Add basic comparison logic here
                response_parts.append(f"Correlation analysis would be needed for detailed comparison.\n\n")
        
        return ''.join(response_parts)
    
    def _generate_temporal_response(self, context: Dict[str, Any]) -> str:
        """Generate temporal analysis response"""
        
        temporal = context.get('data_summary', {}).get('temporal_coverage', {})
        if not temporal:
            return "No temporal data available for analysis."
        
        response_parts = ["Temporal Analysis:\n\n"]
        response_parts.append(f"Time Period: {temporal.get('date_range', ['Unknown', 'Unknown'])[0]} to {temporal.get('date_range', ['Unknown', 'Unknown'])[1]}\n")
        response_parts.append(f"Total Profiles: {temporal.get('total_profiles', 0)}\n")
        response_parts.append(f"Unique Floats: {temporal.get('unique_floats', 0)}\n\n")
        
        return ''.join(response_parts)
    
    def _generate_spatial_response(self, context: Dict[str, Any]) -> str:
        """Generate spatial analysis response"""
        
        geographic = context.get('data_summary', {}).get('geographic_coverage', {})
        if not geographic:
            return "No geographic data available for spatial analysis."
        
        response_parts = ["Spatial Analysis:\n\n"]
        
        lat_range = geographic.get('lat_range', [])
        lon_range = geographic.get('lon_range', [])
        center = geographic.get('center', [])
        
        if lat_range and lon_range:
            response_parts.append(f"Geographic Coverage:\n")
            response_parts.append(f"- Latitude: {lat_range[0]:.2f}° to {lat_range[1]:.2f}°\n")
            response_parts.append(f"- Longitude: {lon_range[0]:.2f}° to {lon_range[1]:.2f}°\n")
            
            if center:
                response_parts.append(f"- Center Point: {center[0]:.2f}°N, {center[1]:.2f}°E\n")
        
        return ''.join(response_parts)
    
    def _generate_general_fallback_response(self, context: Dict[str, Any]) -> str:
        """Generate general fallback response"""
        
        data_summary = context.get('data_summary', {})
        total_records = data_summary.get('total_records', 0)
        
        if total_records == 0:
            return "No relevant oceanographic data found for your query."
        
        response_parts = [f"Found {total_records} relevant oceanographic measurements.\n\n"]
        
        if data_summary.get('available_parameters'):
            response_parts.append(f"Available parameters: {', '.join(data_summary['available_parameters'])}\n")
        
        if context.get('insights'):
            response_parts.append(f"\nKey findings: {', '.join(context['insights'])}")
        
        return ''.join(response_parts)
    
    def _extract_visualization_suggestions(self, query_analysis: Dict[str, Any], 
                                         data: List[Dict[str, Any]]) -> List[str]:
        """Extract visualization suggestions based on query and data"""
        
        suggestions = []
        
        # Based on intent
        intent = query_analysis.get('intent', 'general')
        if intent == 'profile_analysis':
            suggestions.append('depth_profile')
        elif intent == 'spatial_analysis':
            suggestions.append('trajectory_map')
        elif intent == 'temporal_analysis':
            suggestions.append('time_series')
        elif intent == 'comparison':
            suggestions.append('scatter_plot')
        
        # Based on available data
        if data:
            df = pd.DataFrame(data)
            
            if 'depth_m' in df.columns and ('temperature_c' in df.columns or 'salinity_psu' in df.columns):
                suggestions.append('depth_profile')
            
            if 'lat' in df.columns and 'lon' in df.columns:
                suggestions.append('trajectory_map')
            
            if 'profile_datetime' in df.columns:
                suggestions.append('time_series')
            
            # BGC parameters suggest BGC profile
            bgc_params = ['oxygen_umol_kg', 'chlorophyll_mg_m3', 'nitrate_umol_kg', 'ph_total']
            if any(param in df.columns for param in bgc_params):
                suggestions.append('bgc_profile')
        
        # Remove duplicates and return
        return list(set(suggestions))
    
    def _create_error_response(self, query: str, error_msg: str, 
                             filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create error response structure"""
        
        return {
            'query': query,
            'response': f"I encountered an error while processing your oceanographic query: {error_msg}. Please try rephrasing your question or check the data availability.",
            'data': [],
            'processed_context': {'error': error_msg},
            'query_analysis': {'intent': 'error', 'confidence': 0},
            'visualization_suggestions': [],
            'metadata': {
                'retrieved_count': 0,
                'filters_applied': filters or {},
                'error': error_msg,
                'intent_confidence': 0,
                'gemini_response_type': 'error'
            }
        }

# Backward compatibility alias
RAGPipeline = EnhancedRAGPipeline

# Legacy support - keep original class for compatibility
class ArgoRAGPipeline(EnhancedRAGPipeline):
    """Legacy wrapper for backward compatibility"""
    
    def __init__(self, vector_store=None):
        # Use provided vector store or create a simple wrapper
        if vector_store is None:
            from vector_db.vector_store import VectorStore
            vector_store = VectorStore()
        super().__init__(vector_store)