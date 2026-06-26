"""
🧠 ARGO Ocean Data Explorer - Query Orchestration Layer
=======================================================
Phase 11: Intelligent Query Routing and Processing

This module provides intelligent query analysis and routing to determine
the optimal processing pipeline for each user query.

Features:
- LLM-based query intent analysis
- Automatic routing to appropriate processing methods
- Fallback mechanisms for system resilience
- Performance optimization through intelligent caching
- Context-aware processing pipeline selection
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import re

# Import our LLM for query analysis
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries the system can handle."""
    VISUALIZATION_FOCUSED = "visualization_focused"
    DATA_RETRIEVAL = "data_retrieval"
    ANALYTICAL = "analytical"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    SPATIAL_ANALYSIS = "spatial_analysis"
    COMPARATIVE = "comparative"
    GENERAL_QUERY = "general_query"
    SYSTEM_COMMAND = "system_command"

class ProcessingMethod(Enum):
    """Available processing methods."""
    RAG_ONLY = "rag_only"
    MCP_DIRECT = "mcp_direct"
    RAG_PLUS_MCP = "rag_plus_mcp"
    VISUALIZATION_PIPELINE = "visualization_pipeline"
    CACHED_RESPONSE = "cached_response"
    HYBRID_PROCESSING = "hybrid_processing"

class QueryOrchestrator:
    """
    Intelligent query orchestration system that analyzes incoming queries
    and routes them to the optimal processing pipeline.
    """
    
    def __init__(self):
        # Initialize Gemini for query analysis - lazy load to avoid startup errors
        self.llm = None
        self._initialized = False
        
        # Query patterns for quick classification
        self.query_patterns = {
            QueryType.VISUALIZATION_FOCUSED: [
                r'\b(show|plot|graph|visualize|chart|display|map)\b',
                r'\b(temperature profile|t-s diagram|geographic|distribution)\b'
            ],
            QueryType.DATA_RETRIEVAL: [
                r'\b(get|fetch|retrieve|find|data|values|measurements)\b',
                r'\b(temperature|salinity|depth|pressure|float)\b'
            ],
            QueryType.ANALYTICAL: [
                r'\b(analyze|compare|correlation|trend|pattern|statistics)\b',
                r'\b(mean|average|maximum|minimum|variance)\b'
            ],
            QueryType.TEMPORAL_ANALYSIS: [
                r'\b(time|temporal|over time|trends|seasonal|monthly|yearly)\b',
                r'\b(before|after|during|between|since|until)\b'
            ],
            QueryType.SPATIAL_ANALYSIS: [
                r'\b(location|region|area|geographic|spatial|latitude|longitude)\b',
                r'\b(ocean|sea|basin|indian ocean|pacific|atlantic)\b'
            ],
            QueryType.COMPARATIVE: [
                r'\b(compare|versus|vs|difference|similar|different)\b',
                r'\b(higher|lower|warmer|colder|deeper|shallower)\b'
            ]
        }
        
        # Processing method scoring weights
        self.method_weights = {
            QueryType.VISUALIZATION_FOCUSED: {
                ProcessingMethod.VISUALIZATION_PIPELINE: 0.9,
                ProcessingMethod.RAG_PLUS_MCP: 0.7,
                ProcessingMethod.RAG_ONLY: 0.3
            },
            QueryType.DATA_RETRIEVAL: {
                ProcessingMethod.MCP_DIRECT: 0.8,
                ProcessingMethod.RAG_PLUS_MCP: 0.9,
                ProcessingMethod.RAG_ONLY: 0.6
            },
            QueryType.ANALYTICAL: {
                ProcessingMethod.RAG_PLUS_MCP: 0.9,
                ProcessingMethod.HYBRID_PROCESSING: 0.8,
                ProcessingMethod.RAG_ONLY: 0.7
            },
            QueryType.TEMPORAL_ANALYSIS: {
                ProcessingMethod.MCP_DIRECT: 0.9,
                ProcessingMethod.VISUALIZATION_PIPELINE: 0.8,
                ProcessingMethod.RAG_PLUS_MCP: 0.7
            },
            QueryType.SPATIAL_ANALYSIS: {
                ProcessingMethod.VISUALIZATION_PIPELINE: 0.9,
                ProcessingMethod.MCP_DIRECT: 0.8,
                ProcessingMethod.RAG_PLUS_MCP: 0.7
            }
        }
        
        # Cache for query analysis results
        self.analysis_cache: Dict[str, Dict] = {}
        
        logger.info("Query Orchestrator initialized with lazy LLM loading")
    
    def _ensure_llm_initialized(self):
        """Lazy initialization of LLM to avoid startup issues."""
        if not self._initialized:
            try:
                # Load environment variables
                load_dotenv()
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    # Try alternative env var names
                    api_key = os.getenv('GOOGLE_API_KEY')
                
                if not api_key:
                    logger.warning("GEMINI_API_KEY not found - LLM analysis will be disabled")
                    self._initialized = True
                    return False
                
                genai.configure(api_key=api_key)
                self.llm = genai.GenerativeModel('gemini-1.5-flash')
                self._initialized = True
                logger.info("Query Orchestrator LLM initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {str(e)}")
                self._initialized = True
                return False
        
        return self.llm is not None
    
    async def analyze_and_route_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze a query and determine the optimal processing route.
        
        Args:
            query: The natural language query
            context: Optional context from previous interactions
            
        Returns:
            Dictionary containing routing decisions and analysis
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{query.lower().strip()}"
            if cache_key in self.analysis_cache:
                logger.info("Returning cached query analysis")
                return self.analysis_cache[cache_key]
            
            # Step 1: Quick pattern-based classification
            pattern_analysis = self._pattern_based_analysis(query)
            
            # Step 2: LLM-based deep analysis
            llm_analysis = await self._llm_based_analysis(query, context)
            
            # Step 3: Combine analyses and determine routing
            routing_decision = self._determine_optimal_routing(
                query, pattern_analysis, llm_analysis, context
            )
            
            # Step 4: Generate fallback strategies
            fallback_strategies = self._generate_fallback_strategies(routing_decision)
            
            # Compile final analysis
            analysis_result = {
                "query": query,
                "analysis_time": time.time() - start_time,
                "query_type": routing_decision["primary_type"],
                "confidence": routing_decision["confidence"],
                "processing_method": routing_decision["method"],
                "priority_score": routing_decision["priority"],
                "pattern_analysis": pattern_analysis,
                "llm_insights": llm_analysis,
                "fallback_strategies": fallback_strategies,
                "metadata": {
                    "complexity": routing_decision.get("complexity", "medium"),
                    "expected_processing_time": routing_decision.get("estimated_time", 2.0),
                    "resource_requirements": routing_decision.get("resources", ["cpu", "memory"]),
                    "data_sources": routing_decision.get("data_sources", ["rag", "postgresql"])
                }
            }
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis_result
            
            # Limit cache size
            if len(self.analysis_cache) > 50:
                oldest_key = next(iter(self.analysis_cache))
                del self.analysis_cache[oldest_key]
            
            logger.info(f"Query analyzed: {routing_decision['method'].value} route selected with {routing_decision['confidence']:.2f} confidence")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Query analysis error: {str(e)}")
            # Return safe fallback
            return {
                "query": query,
                "analysis_time": time.time() - start_time,
                "query_type": QueryType.GENERAL_QUERY,
                "confidence": 0.5,
                "processing_method": ProcessingMethod.RAG_ONLY,
                "priority_score": 0.5,
                "error": str(e),
                "fallback_strategies": [ProcessingMethod.RAG_ONLY, ProcessingMethod.MCP_DIRECT]
            }
    
    def _pattern_based_analysis(self, query: str) -> Dict[str, Any]:
        """Quick pattern-based query classification using regex."""
        query_lower = query.lower()
        scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    score += len(matches) * 0.2
                    matched_patterns.extend(matches)
            
            if score > 0:
                scores[query_type] = {
                    "score": min(score, 1.0),
                    "matches": matched_patterns
                }
        
        # Determine primary type
        if scores:
            primary_type = max(scores.keys(), key=lambda k: scores[k]["score"])
            confidence = scores[primary_type]["score"]
        else:
            primary_type = QueryType.GENERAL_QUERY
            confidence = 0.3
        
        return {
            "primary_type": primary_type,
            "confidence": confidence,
            "all_scores": scores,
            "method": "pattern_matching"
        }
    
    async def _llm_based_analysis(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Deep LLM-based query analysis for complex intent understanding."""
        
        # Ensure LLM is initialized
        if not self._ensure_llm_initialized():
            logger.warning("LLM not available - falling back to pattern-based analysis")
            return {
                "method": "llm_analysis_unavailable",
                "analysis": {
                    "intent": "general_query",
                    "complexity": "medium", 
                    "confidence": 0.5,
                    "suggested_approach": "rag_based"
                },
                "fallback_used": True
            }
        
        try:
            analysis_prompt = f"""
Analyze this oceanographic data query and provide detailed insights:

Query: "{query}"

Context: {context or "No previous context"}

Please analyze and respond with a JSON object containing:
1. "intent": Primary intent (visualization, data_retrieval, analysis, comparison, etc.)
2. "complexity": Query complexity (simple, medium, complex)
3. "data_requirements": What data is needed
4. "visualization_needed": Boolean - should this include visualizations?
5. "time_sensitivity": Is this time-series related?
6. "spatial_component": Does this involve geographic/spatial analysis?
7. "confidence": Your confidence in this analysis (0-1)
8. "suggested_approach": Recommended processing approach
9. "key_entities": Important entities mentioned (temperature, salinity, location, etc.)
10. "expected_output": What type of output user likely expects

Focus on oceanographic domain expertise. Be concise but thorough.
"""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.generate_content(analysis_prompt)
            )
            
            # Parse LLM response
            llm_text = response.text.strip()
            
            # Try to extract JSON, fallback to text analysis
            try:
                import json
                # Look for JSON in the response
                start_idx = llm_text.find('{')
                end_idx = llm_text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_text = llm_text[start_idx:end_idx]
                    llm_analysis = json.loads(json_text)
                else:
                    raise ValueError("No JSON found")
                    
            except:
                # Fallback: parse manually
                llm_analysis = {
                    "intent": "general_query",
                    "complexity": "medium",
                    "confidence": 0.6,
                    "suggested_approach": "rag_based",
                    "raw_response": llm_text
                }
            
            return {
                "method": "llm_analysis",
                "analysis": llm_analysis,
                "raw_response": llm_text[:500] + "..." if len(llm_text) > 500 else llm_text
            }
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {str(e)}")
            return {
                "method": "llm_analysis_failed",
                "error": str(e),
                "fallback_used": True
            }
    
    def _determine_optimal_routing(
        self, 
        query: str, 
        pattern_analysis: Dict, 
        llm_analysis: Dict, 
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Determine the optimal processing route based on all analyses."""
        
        # Combine pattern and LLM analyses
        primary_type = pattern_analysis["primary_type"]
        pattern_confidence = pattern_analysis["confidence"]
        
        # Adjust based on LLM insights if available
        llm_data = llm_analysis.get("analysis", {})
        if llm_data and "confidence" in llm_data:
            # Weight LLM analysis higher for complex queries
            if llm_data.get("complexity") == "complex":
                llm_weight = 0.7
                pattern_weight = 0.3
            else:
                llm_weight = 0.5
                pattern_weight = 0.5
            
            combined_confidence = (pattern_confidence * pattern_weight + 
                                 llm_data.get("confidence", 0.5) * llm_weight)
        else:
            combined_confidence = pattern_confidence
        
        # Select processing method based on query type and weights
        if primary_type in self.method_weights:
            method_scores = self.method_weights[primary_type]
            optimal_method = max(method_scores.keys(), key=lambda k: method_scores[k])
            method_confidence = method_scores[optimal_method]
        else:
            optimal_method = ProcessingMethod.RAG_ONLY
            method_confidence = 0.6
        
        # Calculate priority score
        priority_factors = {
            "confidence": combined_confidence,
            "method_suitability": method_confidence,
            "complexity_bonus": 0.1 if llm_data.get("complexity") == "complex" else 0,
            "visualization_bonus": 0.1 if llm_data.get("visualization_needed") else 0
        }
        
        priority_score = sum(priority_factors.values()) / len(priority_factors)
        
        # Estimate processing time based on method and complexity
        time_estimates = {
            ProcessingMethod.CACHED_RESPONSE: 0.1,
            ProcessingMethod.RAG_ONLY: 1.5,
            ProcessingMethod.MCP_DIRECT: 2.0,
            ProcessingMethod.RAG_PLUS_MCP: 3.0,
            ProcessingMethod.VISUALIZATION_PIPELINE: 4.0,
            ProcessingMethod.HYBRID_PROCESSING: 5.0
        }
        
        estimated_time = time_estimates.get(optimal_method, 2.0)
        if llm_data.get("complexity") == "complex":
            estimated_time *= 1.5
        
        return {
            "primary_type": primary_type,
            "method": optimal_method,
            "confidence": combined_confidence,
            "priority": priority_score,
            "complexity": llm_data.get("complexity", "medium"),
            "estimated_time": estimated_time,
            "resources": self._estimate_resources(optimal_method),
            "data_sources": self._estimate_data_sources(optimal_method)
        }
    
    def _generate_fallback_strategies(self, routing_decision: Dict) -> List[ProcessingMethod]:
        """Generate fallback strategies if primary method fails."""
        primary_method = routing_decision["method"]
        
        # Define fallback chains
        fallback_chains = {
            ProcessingMethod.VISUALIZATION_PIPELINE: [
                ProcessingMethod.RAG_PLUS_MCP,
                ProcessingMethod.RAG_ONLY
            ],
            ProcessingMethod.RAG_PLUS_MCP: [
                ProcessingMethod.RAG_ONLY,
                ProcessingMethod.MCP_DIRECT
            ],
            ProcessingMethod.HYBRID_PROCESSING: [
                ProcessingMethod.RAG_PLUS_MCP,
                ProcessingMethod.RAG_ONLY
            ],
            ProcessingMethod.MCP_DIRECT: [
                ProcessingMethod.RAG_ONLY
            ],
            ProcessingMethod.RAG_ONLY: [
                ProcessingMethod.MCP_DIRECT
            ]
        }
        
        return fallback_chains.get(primary_method, [ProcessingMethod.RAG_ONLY])
    
    def _estimate_resources(self, method: ProcessingMethod) -> List[str]:
        """Estimate resource requirements for a processing method."""
        resource_map = {
            ProcessingMethod.CACHED_RESPONSE: ["memory"],
            ProcessingMethod.RAG_ONLY: ["cpu", "memory"],
            ProcessingMethod.MCP_DIRECT: ["cpu", "database"],
            ProcessingMethod.RAG_PLUS_MCP: ["cpu", "memory", "database"],
            ProcessingMethod.VISUALIZATION_PIPELINE: ["cpu", "memory", "gpu"],
            ProcessingMethod.HYBRID_PROCESSING: ["cpu", "memory", "database", "gpu"]
        }
        return resource_map.get(method, ["cpu", "memory"])
    
    def _estimate_data_sources(self, method: ProcessingMethod) -> List[str]:
        """Estimate data sources needed for a processing method."""
        source_map = {
            ProcessingMethod.CACHED_RESPONSE: ["cache"],
            ProcessingMethod.RAG_ONLY: ["vector_db"],
            ProcessingMethod.MCP_DIRECT: ["postgresql"],
            ProcessingMethod.RAG_PLUS_MCP: ["vector_db", "postgresql"],
            ProcessingMethod.VISUALIZATION_PIPELINE: ["vector_db", "postgresql", "simulation"],
            ProcessingMethod.HYBRID_PROCESSING: ["vector_db", "postgresql", "simulation", "external_apis"]
        }
        return source_map.get(method, ["vector_db"])
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about query processing patterns."""
        if not self.analysis_cache:
            return {"message": "No queries processed yet"}
        
        analyses = list(self.analysis_cache.values())
        
        # Calculate statistics
        query_types = [a.get("query_type", QueryType.GENERAL_QUERY) for a in analyses]
        methods = [a.get("processing_method", ProcessingMethod.RAG_ONLY) for a in analyses]
        confidences = [a.get("confidence", 0.5) for a in analyses]
        
        from collections import Counter
        
        return {
            "total_queries": len(analyses),
            "most_common_query_types": dict(Counter(query_types).most_common(3)),
            "most_used_methods": dict(Counter(methods).most_common(3)),
            "average_confidence": sum(confidences) / len(confidences),
            "cache_hit_ratio": len([a for a in analyses if a.get("analysis_time", 1) < 0.1]) / len(analyses)
        }

# Global orchestrator instance
query_orchestrator = QueryOrchestrator()