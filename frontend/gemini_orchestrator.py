#!/usr/bin/env python3
"""
Dynamic Gemini-First Orchestration System
Integrates with Gemini API from config.py for intelligent query processing, analysis and response generation.
Removes hardcoded string matching and fixed responses.
"""

import asyncio
import logging
import os
import json
from types import SimpleNamespace
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
from collections import OrderedDict
import plotly.express as px
import plotly.graph_objects as go

# Import configuration
from config import config

try:
    import asyncpg  # Optional; used only if PostgreSQL is enabled
except Exception:  # pragma: no cover
    asyncpg = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiOrchestrator:
    """
    Dynamic Gemini-First Orchestration System
    - Uses Gemini API from config.py for intelligent query analysis
    - No hardcoded string matching or fixed responses
    - Dynamic query classification, SQL decision making, and response generation
    - Coordinates RAG, MCP tools, and data processing intelligently
    """
    
    def __init__(self):
        # Initialize Gemini from config with automatic fallback
        if genai is None:
            raise RuntimeError("google.generativeai not available; install google-generativeai")
        
        # Initialize with API key rotation system
        self._config = config
        self._api_keys = config.API_KEYS
        self._current_key_index = 0
        self._current_api_key = self._api_keys[self._current_key_index]
        self._api_fallback_enabled = config.AUTO_FALLBACK_ON_QUOTA
        
        if not self._api_keys or not self._api_keys[0]:
            raise RuntimeError("No Gemini API keys found in config. Please set API_KEYS")
        
        # Configure Gemini with current API key
        self._configure_gemini_api(self._current_api_key)
        
        logger.info("✅ Gemini Director enabled with dynamic response generation and API fallback")
        
        # Load data
        self.data = self._load_data()
        logger.info(f"✅ Data loaded: {len(self.data)} records")
        
        # Initialize comprehensive MCP server
        self.mcp_server = None
        self._initialize_mcp_server()
        
        # Initialize RAG and MCP components
        self.rag_context = self._initialize_rag_context()
        self.mcp_tools = self._initialize_mcp_tools()

        # Tool execution timeout (seconds)
        self._tool_timeout = float(os.getenv("TOOL_TIMEOUT_SECONDS", "15"))

        # Optional PostgreSQL pool (enable with POSTGRES_DSN or individual env vars)
        self.pg_pool = None
        if os.getenv("USE_POSTGRES", "0") == "1":
            if asyncpg is None:
                logger.warning("⚠️ USE_POSTGRES=1 but asyncpg is not installed. Falling back to in-memory filters.")
            else:
                try:
                    dsn = os.getenv("POSTGRES_DSN")
                    if not dsn:
                        host = os.getenv("PGHOST", "localhost")
                        port = int(os.getenv("PGPORT", "5432"))
                        user = os.getenv("PGUSER", "postgres")
                        password = os.getenv("PGPASSWORD", "password")
                        database = os.getenv("PGDATABASE", "argo_ocean_data")
                        dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
                    # Lazily create pool on first use to avoid blocking init in Streamlit
                    self._pg_dsn = dsn
                except Exception as e:
                    logger.warning(f"⚠️ Failed to configure PostgreSQL: {e}")

        # Simple in-memory LRU cache for heavy visualization results
        self._viz_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._viz_cache_max = int(os.getenv("VIZ_CACHE_MAX", "16"))

    def _configure_gemini_api(self, api_key: str):
        """Configure Gemini API with the given key"""
        try:
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)
            self._use_gemini_director = True
            logger.info(f"✅ Gemini API configured successfully")
        except Exception as e:
            logger.error(f"❌ Failed to configure Gemini API: {e}")
            raise

    def _try_api_fallback(self):
        """Rotate to the next API key if available"""
        if not self._api_fallback_enabled:
            return False
            
        # Calculate next key index
        next_index = (self._current_key_index + 1) % len(self._api_keys)
        
        # If we've cycled through all keys, we can't fallback further
        if next_index == 0 and self._current_key_index != 0:
            logger.error("❌ All API keys have been exhausted")
            return False
        
        # If we're trying the same key, skip rotation
        if next_index == self._current_key_index:
            logger.error("❌ Only one API key available, cannot rotate")
            return False
            
        # Rotate to next key
        self._current_key_index = next_index
        self._current_api_key = self._api_keys[self._current_key_index]
        
        logger.warning(f"⚠️ Switching to API key #{self._current_key_index + 1} due to quota exceeded")
        self._configure_gemini_api(self._current_api_key)
        
        # Update config for other components
        self._config.CURRENT_KEY_INDEX = self._current_key_index
        self._config.GOOGLE_API_KEY = self._current_api_key
        
        return True

    def _handle_api_quota_error(self, error_message: str):
        """Handle API quota errors with automatic key rotation"""
        if "quota" in error_message.lower() or "429" in str(error_message):
            if self._try_api_fallback():
                logger.info(f"✅ Successfully rotated to API key #{self._current_key_index + 1}")
                return True
            else:
                logger.error("❌ No more API keys available for rotation")
                return False
        return False

    def _safe_gemini_call(self, prompt: str, max_retries: int = 2):
        """Make a Gemini API call with automatic key rotation on quota errors"""
        for attempt in range(max_retries + 1):
            try:
                response = self._gemini_model.generate_content(prompt)
                return response
            except Exception as e:
                error_msg = str(e)
                if self._handle_api_quota_error(error_msg) and attempt < max_retries:
                    logger.info(f"Retrying with rotated API key (attempt {attempt + 1}/{max_retries + 1})")
                    continue
                else:
                    raise e

    def get_api_key_status(self):
        """Get current API key rotation status for monitoring"""
        return {
            "current_key_index": self._current_key_index + 1,
            "total_keys": len(self._api_keys),
            "current_key_masked": f"...{self._current_api_key[-8:]}" if self._current_api_key else "None",
            "rotation_enabled": self._api_fallback_enabled
        }

    def _viz_cache_key(self, viz_request: str, filters: Dict[str, Any]) -> str:
        try:
            override = getattr(self, "_query_data_override", None)
            rows = int(len(override)) if override is not None else int(len(self.data))
            filt_str = json.dumps(filters or {}, sort_keys=True)
            return f"{viz_request}|rows={rows}|f={filt_str}"
        except Exception:
            return f"{viz_request}|fallback"

    def _viz_cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._viz_cache:
            self._viz_cache.move_to_end(key)
            return self._viz_cache[key]
        return None

    def _viz_cache_put(self, key: str, value: Dict[str, Any]) -> None:
        self._viz_cache[key] = value
        self._viz_cache.move_to_end(key)
        while len(self._viz_cache) > self._viz_cache_max:
            self._viz_cache.popitem(last=False)

    def _generate_knowledge_based_response(self, user_query: str, context_data: Dict[str, Any] = None) -> str:
        """
        Generate comprehensive responses using Gemini's knowledge when MCP tools fail.
        This replaces basic fallback responses with intelligent, educational content.
        """
        try:
            # Detect the type of query to tailor the response
            query_lower = user_query.lower()
            
            # Check if this is a biological/ecosystem query
            is_biological = any(term in query_lower for term in [
                'phytoplankton', 'bloom', 'algae', 'chlorophyll', 'nutrients', 'nitrogen', 'phosphorus',
                'ecosystem', 'food chain', 'marine biology', 'biodiversity', 'carbon cycle',
                'productivity', 'eutrophication', 'hypoxia', 'dead zone', 'correlation', 'relationship'
            ])
            
            # Check if this is about climate/environmental processes  
            is_climate = any(term in query_lower for term in [
                'climate change', 'global warming', 'ocean acidification', 'sea level',
                'temperature trend', 'warming', 'cooling', 'circulation', 'current'
            ])
            
            # Check if this is about oceanographic processes
            is_oceanographic = any(term in query_lower for term in [
                'upwelling', 'downwelling', 'thermocline', 'halocline', 'density', 'stratification',
                'mixing', 'turbulence', 'front', 'eddy', 'gyre', 'tide', 'wave'
            ])
            
            # Prepare context information if available
            context_info = ""
            if context_data:
                if 'temperature_range' in context_data:
                    context_info += f"\nAvailable temperature data: {context_data['temperature_range']}"
                if 'salinity_range' in context_data:
                    context_info += f"\nAvailable salinity data: {context_data['salinity_range']}"
                if 'region' in context_data:
                    context_info += f"\nRegion of interest: {context_data['region']}"
            
            # Construct specialized prompt based on query type
            if is_biological:
                prompt = f"""
As an expert marine biologist and oceanographer, provide a comprehensive scientific response to: "{user_query}"

Context: The user is working with ARGO oceanographic data but is asking about biological processes.{context_info}

Please provide detailed information covering:
1. The biological mechanisms and processes involved
2. How physical oceanographic parameters (temperature, salinity, density) influence these biological processes
3. What additional data would be needed for such analysis (BGC-ARGO, satellite data, etc.)
4. Regional variations and examples from real-world locations
5. Current scientific understanding and recent research findings
6. Practical implications for marine ecosystems and fisheries

Make this educational, scientifically accurate, and engaging. Include specific examples and mechanisms.
"""
            elif is_climate:
                prompt = f"""
As an expert climate scientist and physical oceanographer, provide a comprehensive response to: "{user_query}"

Context: The user is interested in climate and environmental oceanography.{context_info}

Please cover:
1. The physical processes and mechanisms involved
2. How oceanographic data reveals climate patterns and changes
3. Regional and global implications
4. Time scales involved (seasonal, annual, decadal)
5. Current research and observations
6. Connections to broader Earth system processes

Provide scientifically accurate, detailed information with real-world examples.
"""
            elif is_oceanographic:
                prompt = f"""
As an expert physical oceanographer, explain: "{user_query}"

Context: Focus on oceanographic processes and physical mechanisms.{context_info}

Please explain:
1. The underlying physical processes
2. How these processes affect water properties (temperature, salinity, density)
3. Methods for observing and measuring these phenomena
4. Regional examples and variations
5. Importance for ocean circulation and climate
6. How ARGO and other observational systems detect these features

Provide detailed scientific explanations with practical examples.
"""
            else:
                # General oceanographic query
                prompt = f"""
As an expert oceanographer, provide a comprehensive response to: "{user_query}"

Context: The user is working with oceanographic data and seeks scientific understanding.{context_info}

Please provide:
1. Clear scientific explanations relevant to the question
2. How this relates to oceanographic observations and data
3. Real-world examples and applications
4. Current scientific understanding
5. Practical implications

Make this educational and scientifically accurate.
"""
            
            # Generate response using Gemini
            response = self._safe_gemini_call(prompt)
            
            if response and response.text:
                # Clean and format the response
                knowledge_response = response.text.strip()
                
                # Add a note about data limitations if relevant
                if any(term in query_lower for term in ['correlation', 'analysis', 'compare', 'relationship']):
                    knowledge_response += f"\n\n💡 **Note**: For specific quantitative analysis of this type, specialized datasets beyond standard ARGO measurements would be required."
                
                return knowledge_response
            else:
                return "I can provide information about oceanographic processes, but I'm currently unable to generate a detailed response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Knowledge-based response generation failed: {e}")
            return f"I understand you're asking about {user_query}, but I'm currently unable to provide a detailed response. Please try again or rephrase your question."

    # ------------------------
    # Utility: Filters & Query
    # ------------------------
    async def _classify_query_type(self, user_query: str) -> str:
        """Use Gemini to intelligently classify query type instead of hardcoded rules."""
        try:
            prompt = f"""
            Classify this user query into one of these categories:
            - "conversational": Greetings, thanks, general chat (e.g., "hi", "hello", "thank you")
            - "informational": Questions about concepts, definitions, explanations (e.g., "what is thermocline?", "explain salinity")
            - "analytical": Data analysis requests requiring computation (e.g., "average temperature", "highest salinity", "compare regions")
            
            Query: "{user_query}"
            
            Respond with only the category name.
            """
            
            response = self._safe_gemini_call(prompt)
            classification = response.text.strip().lower()
            
            if classification in ["conversational", "informational", "analytical"]:
                return classification
            else:
                # Default to analytical for data-related queries
                return "analytical"
                
        except Exception as e:
            logger.warning(f"Query classification failed, defaulting to analytical: {e}")
            return "analytical"

    def _apply_filters(self, df: pd.DataFrame, filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Enhanced filtering with proper region mapping and logging."""
        if df is None or df.empty or not filters:
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

        out = df.copy()
        original_count = len(out)
        
        try:
            logger.info(f"🔍 Starting filter application. Original data: {original_count} records")
            logger.info(f"🔍 Filters to apply: {filters}")
            
            # Ocean gating: dataset is Indian Ocean; fail fast for other oceans
            ocean = str(filters.get("ocean", "")).strip().lower()
            if ocean and ocean not in {"indian", "indian ocean"}:
                logger.info(f"❌ Ocean filter mismatch: requested {ocean}, data is Indian Ocean")
                return out.iloc[0:0]

            # ENHANCED REGION FILTERING with explicit mapping
            region = str(filters.get("region", "")).strip().lower()
            if region and "latitude" in out.columns:
                logger.info(f"🌊 Applying region filter: {region}")
                
                if region in ["indian", "indian ocean"]:
                    # Indian Ocean specific bounds from actual data
                    out = out[
                        (out["latitude"] >= -12.37) & (out["latitude"] <= -8.05) &
                        (out["longitude"] >= 39.72) & (out["longitude"] <= 55.95)
                    ]
                    logger.info(f"🌊 Indian Ocean filter applied: {len(out)} records remain")
                elif region in ["equator", "equatorial"]:
                    out = out[(out["latitude"] >= -5) & (out["latitude"] <= 5)]
                    logger.info(f"🌍 Equatorial filter applied: {len(out)} records remain")
                elif region in ["tropical", "tropics"]:
                    out = out[(out["latitude"] >= -23.5) & (out["latitude"] <= 23.5)]
                    logger.info(f"🏝️ Tropical filter applied: {len(out)} records remain")
                elif region in ["arctic", "north polar"]:
                    out = out[out["latitude"] >= 66.5]
                elif region in ["antarctic", "south polar"]:
                    out = out[out["latitude"] <= -66.5]
                elif region in ["northern hemisphere", "north"]:
                    out = out[out["latitude"] >= 0]
                elif region in ["southern hemisphere", "south"]:
                    out = out[out["latitude"] <= 0]

            # EXPLICIT LAT/LON RANGE FILTERING (overrides region if specified)
            lat_range = filters.get("lat_range") or filters.get("latitude_range")
            if lat_range and len(lat_range) == 2 and "latitude" in out.columns:
                lo, hi = float(lat_range[0]), float(lat_range[1])
                out = out[(out["latitude"] >= lo) & (out["latitude"] <= hi)]
                logger.info(f"📍 Latitude range [{lo}, {hi}] applied: {len(out)} records remain")

            lon_range = filters.get("lon_range") or filters.get("longitude_range")
            if lon_range and len(lon_range) == 2 and "longitude" in out.columns:
                lo, hi = float(lon_range[0]), float(lon_range[1])
                out = out[(out["longitude"] >= lo) & (out["longitude"] <= hi)]
                logger.info(f"📍 Longitude range [{lo}, {hi}] applied: {len(out)} records remain")

            # Year filtering
            year_filter = filters.get("year") or filters.get("years")
            if year_filter is not None and "date" in out.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(out["date"]):
                        out = out.copy()
                        out["date"] = pd.to_datetime(out["date"], errors="coerce")
                    
                    if isinstance(year_filter, (int, str)):
                        year_filter = [int(year_filter)]
                    elif isinstance(year_filter, (list, tuple)):
                        year_filter = [int(y) for y in year_filter]
                    
                    out = out[out["date"].dt.year.isin(year_filter)]
                    logger.info(f"📅 Year filter applied: {len(out)} records remain")
                except Exception as e:
                    logger.warning(f"⚠️ Year filtering failed: {e}")

            # Depth/pressure filtering
            depth_range = filters.get("depth_range") or filters.get("pressure_range")
            if depth_range and len(depth_range) == 2:
                lo, hi = float(depth_range[0]), float(depth_range[1])
                depth_col = "depth" if "depth" in out.columns else "pressure"
                if depth_col in out.columns:
                    out = out[(out[depth_col] >= lo) & (out[depth_col] <= hi)]
                    logger.info(f"📊 {depth_col.capitalize()} range [{lo}, {hi}] applied: {len(out)} records remain")

            # Final logging
            if out.empty:
                logger.warning(f"❌ ALL DATA FILTERED OUT! Original: {original_count}, Final: 0")
                logger.warning(f"❌ Problematic filters: {filters}")
            else:
                logger.info(f"✅ Filtering complete: {original_count} → {len(out)} records ({100*len(out)/original_count:.1f}%)")
            
            return out
            
        except Exception as e:
            logger.error(f"❌ Filter application failed: {e}")
            return df

            # Month filtering
            month_filter = filters.get("month") or filters.get("months")
            if month_filter is not None and "date" in out.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(out["date"]):
                        out = out.copy()
                        out["date"] = pd.to_datetime(out["date"], errors="coerce")
                    
                    if isinstance(month_filter, (int, str)):
                        month_filter = [int(month_filter)]
                    elif isinstance(month_filter, (list, tuple)):
                        month_filter = [int(m) for m in month_filter]
                    
                    out = out[out["date"].dt.month.isin(month_filter)]
                    logger.info(f"📅 Month filter applied: {len(out)} records remain")
                except Exception as e:
                    logger.warning(f"⚠️ Month filtering failed: {e}")

            # Time range filtering
            tr = filters.get("time_range") or filters.get("date_range")
            if tr and len(tr) == 2 and all(tr):
                try:
                    start = pd.to_datetime(tr[0])
                    end = pd.to_datetime(tr[1])
                    if "date" in out.columns:
                        if not pd.api.types.is_datetime64_any_dtype(out["date"]):
                            out = out.copy()
                            out["date"] = pd.to_datetime(out["date"], errors="coerce")
                        out = out[(out["date"] >= start) & (out["date"] <= end)]
                        logger.info(f"📅 Time range filter applied: {len(out)} records remain")
                except Exception as e:
                    logger.warning(f"⚠️ Time range filtering failed: {e}")

            # Float ID filtering
            fids = filters.get("float_ids") or filters.get("float_id")
            if fids is not None:
                if not isinstance(fids, (list, tuple, set)):
                    fids = [fids]
                col = "float_id" if "float_id" in out.columns else ("profile_index" if "profile_index" in out.columns else None)
                if col:
                    out = out[out[col].isin(list(fids))]
                    logger.info(f"🔢 Float ID filter applied: {len(out)} records remain")

            # Profile index filtering
            pidx = filters.get("profile_indices") or filters.get("profile_index")
            if pidx is not None:
                if not isinstance(pidx, (list, tuple, set)):
                    pidx = [pidx]
                if "profile_index" in out.columns:
                    out = out[out["profile_index"].isin(list(pidx))]
                    logger.info(f"🔢 Profile index filter applied: {len(out)} records remain")
                    
        except Exception as e:
            logger.error(f"❌ Filter application failed: {e}")
            return df

    def _apply_filters_with_fallback(self, df: pd.DataFrame, filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Enhanced filtering with progressive fallback strategy."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        original_count = len(df)
        logger.info(f"🔍 Pipeline validation: Starting with {original_count} records")
        
        if not filters:
            logger.info("✅ No filters specified, returning full dataset")
            return df
        
        # Progressive filtering strategy
        strategies = [
            ("Exact Filters", filters),
            ("Broadened Geographic", self._broaden_geographic_filters(filters, 1.5)),
            ("Relaxed Filters", self._broaden_geographic_filters(filters, 2.0)),
            ("Available Data", {})  # No filters - use all available data
        ]
        
        for strategy_name, strategy_filters in strategies:
            try:
                filtered_data = self._apply_filters(df, strategy_filters)
                
                if not filtered_data.empty:
                    success_rate = len(filtered_data) / original_count * 100
                    logger.info(f"✅ {strategy_name}: {len(filtered_data)} records ({success_rate:.1f}%)")
                    return filtered_data
                else:
                    logger.warning(f"⚠️ {strategy_name}: No data found, trying next strategy")
                    
            except Exception as e:
                logger.error(f"❌ {strategy_name} failed: {e}")
                continue
        
        # Final fallback - return original data
        logger.warning(f"🚨 All filtering strategies failed, returning original {original_count} records")
        return df

    def _broaden_geographic_filters(self, filters: Dict[str, Any], factor: float) -> Dict[str, Any]:
        """Broaden geographic filters by expanding lat/lon ranges."""
        if not filters:
            return {}
            
        broadened = filters.copy()
        
        # Expand latitude range
        if "lat_range" in filters and filters["lat_range"]:
            lat_min, lat_max = filters["lat_range"]
            lat_center = (lat_min + lat_max) / 2
            lat_span = (lat_max - lat_min) * factor / 2
            broadened["lat_range"] = [lat_center - lat_span, lat_center + lat_span]
            
        # Expand longitude range  
        if "lon_range" in filters and filters["lon_range"]:
            lon_min, lon_max = filters["lon_range"]
            lon_center = (lon_min + lon_max) / 2
            lon_span = (lon_max - lon_min) * factor / 2
            broadened["lon_range"] = [lon_center - lon_span, lon_center + lon_span]
            
        return broadened

    def _make_json_safe(self, obj: Any) -> Any:
        """Recursively convert numpy/pandas types to JSON-serializable Python types."""
        try:
            import numpy as np
        except Exception:
            np = None

        if obj is None:
            return None
        if isinstance(obj, (str, bool, int, float)):
            return obj
        # numpy scalar
        if np is not None and isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        # pandas Timestamp or datetime
        if hasattr(obj, "isoformat"):
            try:
                return obj.isoformat()
            except Exception:
                pass
        # dict
        if isinstance(obj, dict):
            return {str(self._make_json_safe(k)): self._make_json_safe(v) for k, v in obj.items()}
        # list/tuple
        if isinstance(obj, (list, tuple, set)):
            return [self._make_json_safe(v) for v in obj]
        # fallback to string
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return str(obj)

    
        
    def _load_data(self):
        """Load ARGO data from CSV"""
        try:
            data = pd.read_csv('data/1900121_prof.csv')
            logger.info(f"✅ Loaded {len(data)} ARGO measurements")
            return data
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            # Fallback to synthetic sample data so the app remains usable
            try:
                import numpy as np
                np.random.seed(42)
                
                # Real ARGO float coordinates from the actual dataset
                real_coordinates = [
                    (-12.368, 47.239), (-12.335, 47.467), (-12.162, 47.161), (-12.083, 47.147),
                    (-11.998, 47.161), (-11.861, 46.745), (-11.834, 46.818), (-11.782, 47.207),
                    (-11.775, 47.377), (-11.759, 46.567), (-11.756, 47.298), (-11.729, 46.642),
                    (-11.705, 47.382), (-11.565, 47.577), (-11.534, 47.242), (-11.469, 48.363),
                    (-11.456, 47.485), (-11.309, 47.268), (-11.298, 48.441), (-11.295, 44.251),
                    (-11.270, 48.626), (-11.254, 47.419), (-11.151, 41.411), (-11.113, 44.803),
                    (-11.049, 46.409), (-11.047, 45.519), (-11.039, 43.669), (-10.958, 41.303),
                    (-10.918, 47.340), (-10.838, 44.305), (-10.768, 48.779), (-10.658, 42.343),
                    (-10.644, 41.592), (-10.635, 44.602), (-10.621, 42.856), (-10.535, 44.277),
                    (-10.500, 48.909), (-10.480, 54.816), (-10.462, 54.622), (-10.396, 41.922),
                    (-10.326, 54.926), (-10.326, 42.032), (-10.321, 49.139), (-10.318, 54.519),
                    (-10.286, 54.846), (-10.281, 41.028), (-10.250, 54.724), (-10.232, 54.476),
                    (-10.224, 54.390), (-10.221, 54.957), (-10.169, 40.855), (-10.158, 54.561),
                    (-10.151, 54.768), (-10.145, 54.781), (-10.144, 40.745), (-10.114, 54.538),
                    (-10.105, 55.066), (-10.100, 40.879), (-10.093, 40.764), (-10.059, 55.142),
                    (-10.057, 40.944), (-10.054, 54.320), (-10.024, 55.666), (-9.989, 55.202),
                    (-9.987, 40.709), (-9.986, 55.492), (-9.976, 49.439), (-9.925, 55.695),
                    (-9.918, 40.713), (-9.912, 54.050), (-9.875, 40.533), (-9.857, 55.953),
                    (-9.825, 55.439), (-9.767, 50.009), (-9.746, 50.338), (-9.743, 55.198),
                    (-9.741, 40.353), (-9.712, 55.321), (-9.687, 49.661), (-9.638, 53.872),
                    (-9.605, 40.155), (-9.554, 53.351), (-9.424, 51.921), (-9.414, 40.070),
                    (-9.383, 52.278), (-9.374, 40.057), (-9.357, 52.882), (-9.352, 53.179),
                    (-9.329, 51.539), (-9.278, 51.532), (-9.265, 51.970), (-9.211, 51.855),
                    (-9.185, 51.276), (-9.121, 50.968), (-9.096, 51.515), (-9.089, 39.889),
                    (-8.747, 39.816), (-8.460, 39.717), (-8.046, 39.820)
                ]
                
                n_coordinates = len(real_coordinates)
                n_measurements = 4000
                
                # Create multiple measurements per coordinate location
                measurements_per_coord = n_measurements // n_coordinates
                extra_measurements = n_measurements % n_coordinates
                
                latitudes = []
                longitudes = []
                
                for i, (lat, lon) in enumerate(real_coordinates):
                    # Add base measurements for this coordinate
                    count = measurements_per_coord
                    # Distribute extra measurements among first few coordinates
                    if i < extra_measurements:
                        count += 1
                    
                    # Add slight random variation around actual coordinates to simulate multiple profiles
                    lat_variation = np.random.normal(0, 0.01, count)  # Small variation
                    lon_variation = np.random.normal(0, 0.01, count)
                    
                    latitudes.extend([lat + var for var in lat_variation])
                    longitudes.extend([lon + var for var in lon_variation])
                
                data = pd.DataFrame({
                    'profile_index': np.random.choice(range(1000, 1500), n_measurements),
                    'float_id': np.random.choice(range(1900000, 1900100), n_measurements),
                    'latitude': latitudes[:n_measurements],  # Use real coordinates
                    'longitude': longitudes[:n_measurements],  # Use real coordinates
                    'temperature': np.random.uniform(2.3, 30.33, n_measurements),
                    'salinity': np.random.uniform(34.1, 35.62, n_measurements),
                    'pressure': np.random.uniform(5.5, 1999.9, n_measurements),
                    'depth': np.random.uniform(0, 2000, n_measurements),
                    'date': pd.date_range('2020-01-01', periods=n_measurements, freq='D')
                })
                logger.info(f"🧪 Using synthetic dataset with {len(data)} rows and REAL ARGO coordinates (CSV missing)")
                return data
            except Exception as e2:
                logger.error(f"❌ Failed to create synthetic data: {e2}")
            return pd.DataFrame()
    
    def _initialize_mcp_server(self):
        """Initialize comprehensive MCP server with visualization capabilities"""
        try:
            # Try to import and initialize the comprehensive MCP server
            from mcp.mcp_server import ArgoMCPServer
            self.mcp_server = ArgoMCPServer()
            logger.info("✅ Comprehensive MCP server initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Could not import comprehensive MCP server: {e}")
            logger.warning("⚠️ Using basic MCP tools only")
            self.mcp_server = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP server: {e}")
            self.mcp_server = None
    
    def _initialize_rag_context(self):
        """Initialize RAG context with oceanographic knowledge"""
        return {
            "argo_program": "ARGO is a global array of autonomous profiling floats that measure temperature, salinity, and pressure in the upper 2000m of the ocean.",
            "indian_ocean": "The Indian Ocean is characterized by seasonal monsoon winds, upwelling regions, and complex circulation patterns.",
            "temperature_fronts": "Ocean fronts are narrow zones where water properties change rapidly, often associated with strong currents and mixing.",
            "argo_float_1900121": "This float operated in the Indian Ocean between 2002-2005, providing valuable data on regional oceanographic conditions."
        }
    
    def _initialize_mcp_tools(self):
        """Initialize comprehensive MCP tools for oceanographic analysis"""
        # Basic analysis and plotting tools
        tools = {
            "plot_temperature_map": self._plot_temperature_map,
            "plot_salinity_profile": self._plot_salinity_profile,
            "analyze_thermocline": self._analyze_thermocline,
            "calculate_statistics": self._calculate_statistics,
            "filter_spatial_data": self._filter_spatial_data,
            "plot_depth_profile": self._plot_depth_profile,
            "temperature_salinity_scatter": self._plot_temp_sal_scatter,
            "max_temperature": self._compute_max_temperature,
            "max_salinity": self._compute_max_salinity,
            # Generic plotting tools
            "plot_scatter": self._plot_generic_scatter,
            "plot_line": self._plot_generic_line,
            "plot_histogram": self._plot_generic_histogram,
            "plot_xy": self._plot_generic_xy,
            "create_plot": self._create_generic_plot,
            # Aliases from LLM plans
            "thermocline_analysis": self._analyze_thermocline,
            "statistics_summary": self._calculate_statistics
        }
        
        # Add comprehensive MCP server tools if available
        if self.mcp_server:
            # Visualization tools from MCP server (only tools that actually exist)
            comprehensive_tools = {
                'generate_temperature_profile_plot': self._wrap_mcp_tool('generate_temperature_profile_plot'),
                'generate_geographic_map': self._wrap_mcp_tool('generate_geographic_map'),
                'generate_time_series_plot': self._wrap_mcp_tool('generate_time_series_plot'),
                'generate_correlation_matrix': self._wrap_mcp_tool('generate_correlation_matrix'),
                
                # Analysis tools from MCP server
                'get_float_trajectory': self._wrap_mcp_tool('get_float_trajectory'),
                'analyze_float_performance': self._wrap_mcp_tool('analyze_float_performance'),
                'get_profile_details': self._wrap_mcp_tool('get_profile_details'),
                'analyze_depth_profile': self._wrap_mcp_tool('analyze_depth_profile'),
                'get_regional_statistics': self._wrap_mcp_tool('get_regional_statistics'),
                'assess_data_quality': self._wrap_mcp_tool('assess_data_quality'),
            }
            tools.update(comprehensive_tools)
            logger.info(f"✅ Added {len(comprehensive_tools)} comprehensive MCP tools")
        
        logger.info(f"📊 Total MCP tools registered: {len(tools)}")
        return tools
        
    def _wrap_mcp_tool(self, tool_name: str):
        """Wrap MCP server tools for use in orchestrator"""
        async def wrapper(data_or_params):
            try:
                if not self.mcp_server:
                    return {
                        'ok': False,
                        'error': 'MCP server not available',
                        'type': tool_name
                    }
                
                # Initialize MCP server if needed
                if not hasattr(self.mcp_server, 'db_pool') or not self.mcp_server.db_pool:
                    await self.mcp_server.initialize()
                
                # Convert data to parameters if needed
                if isinstance(data_or_params, pd.DataFrame):
                    # Convert DataFrame to parameter dict for MCP server
                    if tool_name == 'generate_temperature_profile_plot':
                        # Extract profile IDs from data
                        profile_ids = data_or_params['profile_index'].unique()[:5] if 'profile_index' in data_or_params.columns else []
                        params = {
                            'profile_ids': [str(pid) for pid in profile_ids],
                            'comparison_mode': len(profile_ids) > 1
                        }
                    elif tool_name == 'generate_geographic_map':
                        # Extract float IDs for geographic mapping
                        float_ids = data_or_params['float_id'].unique()[:10] if 'float_id' in data_or_params.columns else []
                        params = {
                            'float_ids': [str(fid) for fid in float_ids],
                            'parameter': 'temperature',
                            'time_range': None
                        }
                    elif tool_name in ['get_float_trajectory', 'analyze_float_performance']:
                        # Use first available float ID
                        float_ids = data_or_params['float_id'].unique() if 'float_id' in data_or_params.columns else []
                        if len(float_ids) > 0:
                            params = {'float_id': str(float_ids[0])}
                        else:
                            return {'ok': False, 'error': 'No float IDs available in data'}
                    else:
                        # Generic conversion
                        params = {'data_summary': f"{len(data_or_params)} records"}
                else:
                    # Already in parameter format
                    params = data_or_params or {}
                
                # Execute the tool
                result = await self.mcp_server.execute_tool(tool_name, params)
                
                if result.get('success'):
                    return {
                        'ok': True,
                        'type': tool_name,
                        'result': result['result'],
                        'timestamp': result.get('timestamp'),
                        'description': f"Generated {tool_name}"
                    }
                else:
                    return {
                        'ok': False,
                        'error': result.get('error', 'Unknown error'),
                        'type': tool_name
                    }
                    
            except Exception as e:
                logger.error(f"❌ MCP tool {tool_name} failed: {e}")
                return {
                    'ok': False,
                    'error': str(e),
                    'type': tool_name
                }
        
        return wrapper

    # --- PostgreSQL safe query executor ---
    async def _ensure_pg_pool(self):
        if asyncpg is None:
            raise RuntimeError("asyncpg not installed")
        if getattr(self, "pg_pool", None) is None:
            self.pg_pool = await asyncpg.create_pool(dsn=self._pg_dsn, min_size=1, max_size=4)
        return self.pg_pool

    async def _execute_safe_query(self, db_query: Dict[str, Any], row_limit: int = 50000, timeout: int = 10) -> pd.DataFrame:
        WHITELIST_TABLES = {"profiles", "measurements", "floats", "argo_profiles"}
        COLUMN_WHITELIST = {
            "profiles": ["profile_id","float_id","lat","lon","temperature_c","salinity_psu","pressure_dbar","date"],
            "measurements": ["profile_id","depth_m","temperature_c","salinity_psu","pressure_dbar","date"],
            "floats": ["float_id","wmo","lat","lon","date"],
            "argo_profiles": ["float_id","profile_index","latitude","longitude","temperature","salinity","pressure","depth","date"],
        }

        table = str(db_query.get("table", ""))
        columns = list(db_query.get("columns", []))
        filters = dict(db_query.get("filters", {}))
        time_range = db_query.get("time_range")
        aggregations = list(db_query.get("aggregations", []))
        group_by = list(db_query.get("group_by", []))
        order_by = list(db_query.get("order_by", []))

        if table not in WHITELIST_TABLES:
            raise ValueError(f"Table {table} is not whitelisted")
        # Validate selected columns and aggregation columns
        if not columns and not aggregations:
            raise ValueError("columns or aggregations must be provided")
        for col in columns:
            if col not in COLUMN_WHITELIST.get(table, []):
                raise ValueError(f"Column {col} not allowed for {table}")
        for agg in aggregations:
            col = agg.get("column")
            if col and col not in COLUMN_WHITELIST.get(table, []):
                raise ValueError(f"Aggregation column {col} not allowed for {table}")

        # Build safe WHERE with validated filter keys and limited operators
        where_sql = ["TRUE"]
        params = []
        idx = 1
        allowed_cols = set(COLUMN_WHITELIST.get(table, []))

        # Translate pseudo-filters (region, lat_range, lon_range, year)
        region = str(filters.pop("region", "")).strip().lower() if "region" in filters else ""
        lat_range = filters.pop("lat_range", None)
        lon_range = filters.pop("lon_range", None)
        year_value = filters.pop("year", None)

        if isinstance(lat_range, (list, tuple)) and len(lat_range) == 2:
            where_sql.append(f"latitude BETWEEN ${idx} AND ${idx+1}")
            params.extend([float(lat_range[0]), float(lat_range[1])])
            idx += 2
        if isinstance(lon_range, (list, tuple)) and len(lon_range) == 2:
            where_sql.append(f"longitude BETWEEN ${idx} AND ${idx+1}")
            params.extend([float(lon_range[0]), float(lon_range[1])])
            idx += 2
        if year_value is not None:
            where_sql.append(f"DATE_PART('year', date) = ${idx}")
            params.append(int(year_value))
            idx += 1

        for key, val in filters.items():
            if key not in allowed_cols:
                raise ValueError(f"Filter column {key} not allowed for {table}")
            # Support simple value equality or structured op dict
            if isinstance(val, dict):
                op = str(val.get("op", "eq")).lower()
                if op == "eq":
                    where_sql.append(f"{key} = ${idx}")
                    params.append(val.get("value"))
                    idx += 1
                elif op == "gte":
                    where_sql.append(f"{key} >= ${idx}")
                    params.append(val.get("value"))
                    idx += 1
                elif op == "lte":
                    where_sql.append(f"{key} <= ${idx}")
                    params.append(val.get("value"))
                    idx += 1
                elif op == "between":
                    lo, hi = val.get("low"), val.get("high")
                    where_sql.append(f"{key} BETWEEN ${idx} AND ${idx+1}")
                    params.extend([lo, hi])
                    idx += 2
                else:
                    raise ValueError(f"Operator {op} not allowed")
            else:
                where_sql.append(f"{key} = ${idx}")
                params.append(val)
                idx += 1
        if time_range and len(time_range) == 2:
            where_sql.append(f"date BETWEEN ${idx} AND ${idx+1}")
            params.extend(time_range)
            idx += 2

        # Build SELECT with aggregations and group by
        select_parts = []
        if columns:
            select_parts.extend(columns)
        for agg in aggregations:
            op = str(agg.get("op", "")).upper()
            col = agg.get("column")
            alias = agg.get("as", f"{op.lower()}_{col}")
            if op not in {"AVG","SUM","MIN","MAX","COUNT"}:
                raise ValueError(f"Aggregation {op} not allowed")
            select_parts.append(f"{op}({col}) AS {alias}")
        if not select_parts:
            raise ValueError("Nothing to SELECT")

        group_clause = ""
        if group_by:
            # Only allow simple expressions referencing whitelisted columns and basic YEAR(date)
            safe_groups = []
            for g in group_by:
                g = str(g)
                if g.startswith("YEAR(") and ")" in g:
                    inner = g[g.find("(")+1:g.rfind(")")].strip()
                    if inner != "date":
                        raise ValueError("Only YEAR(date) is allowed")
                    safe_groups.append("DATE_PART('year', date)")
                elif g in COLUMN_WHITELIST.get(table, []):
                    safe_groups.append(g)
                else:
                    raise ValueError(f"Group by expression not allowed: {g}")
            group_clause = f" GROUP BY {', '.join(safe_groups)}"
            # If grouping by year, ensure year projection is present for plotting
            if any(sg.startswith("DATE_PART('year',") for sg in safe_groups):
                if not any(" AS year" in part or part.strip()=="year" for part in select_parts):
                    select_parts.insert(0, "DATE_PART('year', date) AS year")

        order_clause = ""
        if order_by:
            safe_orders = []
            for ob in order_by:
                col = ob.get("column")
                direction = str(ob.get("direction","ASC")).upper()
                if direction not in {"ASC","DESC"}:
                    raise ValueError("Invalid order direction")
                # allow ordering by selected columns or aliases like year
                if (col in columns) or any(col == (a.get("as") or f"{a.get('op','').lower()}_{a.get('column')}") for a in aggregations) or str(col).lower() in {"year"}:
                    safe_orders.append(f"{col} {direction}")
                else:
                    raise ValueError(f"Order by column not allowed: {col}")
            if safe_orders:
                order_clause = f" ORDER BY {', '.join(safe_orders)}"

        sql = f"SELECT {', '.join(select_parts)} FROM {table} WHERE {' AND '.join(where_sql)}{group_clause}{order_clause} LIMIT {int(row_limit)}"
        pool = await self._ensure_pg_pool()
        async with pool.acquire() as conn:
            rows = await asyncio.wait_for(conn.fetch(sql, *params), timeout=timeout)
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([dict(r) for r in rows])
    
    def _requires_data_analysis(self, user_query: str) -> bool:
        """Deprecated: Gemini decides if tools/SQL are needed."""
        return False

    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Dynamic Gemini-First Orchestration Method
        1. Gemini classifies the query type (conversational/informational/analytical)
        2. Routes to appropriate handler - Gemini generates dynamic responses
        3. For analytical queries: Gemini Director -> Task Executor -> Gemini Summarizer
        4. No hardcoded responses - all answers generated by Gemini
        """
        try:
            logger.info(f"🔍 Processing query: {user_query}")
            
            # Step 1: Use Gemini to classify query type
            query_type = await self._classify_query_type(user_query)
            logger.info(f"📊 Query classified as: {query_type}")
            
            # Step 2: Handle based on classification
            if query_type == "conversational":
                return await self._handle_conversational_query(user_query)
            elif query_type == "informational":
                return await self._handle_informational_query(user_query)
            else:  # analytical
                return await self._handle_analytical_query(user_query)
                
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            # Even error responses should be generated by Gemini
            try:
                error_prompt = f"""
                There was an error processing this user query: "{user_query}"
                Error: {str(e)}
                
                Provide a helpful, friendly response explaining that there was a technical issue 
                and suggest they try rephrasing their question or asking something simpler.
                Keep it brief and encouraging.
                """
                response = self._safe_gemini_call(error_prompt)
                return {
                    "response": response.text.strip(),
                    "method": "error_handled_by_gemini",
                    "error": str(e)
                }
            except:
                return {
                    "response": "I encountered a technical issue. Please try rephrasing your question.",
                    "method": "fallback_error",
                    "error": str(e)
                }
    
    async def _handle_conversational_query(self, user_query: str) -> Dict[str, Any]:
        """Handle conversational queries with dynamic Gemini responses."""
        try:
            prompt = f"""
            The user said: "{user_query}"
            
            This appears to be a conversational message. Respond in a friendly, helpful way.
            Let them know you're here to help analyze oceanographic data and suggest some example questions they could ask.
            
            Keep your response warm, brief, and include 2-3 specific example questions about:
            - Ocean temperature analysis
            - Salinity patterns  
            - Geographic comparisons
            - Statistical summaries
            
            Make it natural and encouraging.
            """
            
            response = self._safe_gemini_call(prompt)
            return {
                "response": response.text.strip(),
                "method": "conversational_gemini",
                "query_type": "conversational"
            }
            
        except Exception as e:
            logger.error(f"❌ Conversational query handling failed: {e}")
            return {
                "response": "Hello! I'm here to help you analyze oceanographic data. You can ask me about temperature patterns, salinity distributions, or statistical summaries.",
                "method": "conversational_fallback",
                "error": str(e)
            }
    
    async def _handle_informational_query(self, user_query: str) -> Dict[str, Any]:
        """Handle informational queries with Gemini + RAG context."""
        try:
            # Try RAG pipeline first, then Gemini for general knowledge
            rag_response = None
            try:
                # Load RAG pipeline if available
                from rag.rag_pipeline import ArgoRAGPipeline
                rag = ArgoRAGPipeline()
                rag_response = rag.process_query(user_query)
            except Exception as rag_error:
                logger.warning(f"RAG pipeline failed: {rag_error}")
            
            # Use Gemini to provide informational response
            query_lower = user_query.lower()
            is_biological_query = any(term in query_lower for term in [
                'phytoplankton', 'bloom', 'algae', 'chlorophyll', 'nutrients', 'nitrogen', 'phosphorus',
                'ecosystem', 'food chain', 'marine biology', 'biodiversity', 'carbon cycle',
                'climate change', 'global warming', 'ocean acidification', 'coral', 'fish',
                'productivity', 'eutrophication', 'hypoxia', 'dead zone'
            ])
            
            if is_biological_query:
                # Use comprehensive knowledge-based response for biological questions
                logger.info("🧠 Using knowledge-based response for biological informational query")
                response_text = await self._knowledge_based_response(user_query, f"Available Data: ARGO temperature, salinity, and pressure measurements\nRAG Context: {rag_response or 'None'}")
                return {
                    "response": response_text,
                    "method": "knowledge_based_informational",
                    "query_type": "informational_biological",
                    "rag_available": rag_response is not None
                }
            
            prompt = f"""
            The user asked: "{user_query}"
            
            This is an informational question about oceanography or related concepts.
            
            """
            
            if rag_response:
                prompt += f"Available context from knowledge base: {rag_response}\n\n"
            
            prompt += """
            Provide a clear, informative explanation. Focus on:
            - Clear definitions and concepts
            - How it relates to oceanographic data analysis
            - Why it's important for understanding ocean data
            
            Keep it educational but accessible. Limit to 2-3 paragraphs.
            """
            
            response = self._safe_gemini_call(prompt)
            return {
                "response": response.text.strip(),
                "method": "informational_gemini",
                "query_type": "informational",
                "rag_available": rag_response is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Informational query handling failed: {e}")
            return {
                "response": "I can help explain oceanographic concepts. Could you be more specific about what you'd like to learn?",
                "method": "informational_fallback",
                "error": str(e)
            }
    
    async def _handle_analytical_query(self, user_query: str) -> Dict[str, Any]:
        """Handle analytical queries requiring data analysis."""
        try:
            logger.info("🎯 Processing analytical query with Gemini Director")

            # Step 1: Gemini Director - create analysis plan
            instructions = await self._gemini_director(user_query)
            logger.info(f"📋 Gemini generated instructions: {instructions}")

            # Step 2: Validation
            self._validate_instructions(user_query, instructions)

            # Step 3: If Gemini says no tools needed and provides an answer, return it
            if not instructions.get("needs_db_query") and not instructions.get("visualization_requests") and not instructions.get("analysis_requests"):
                ua = (instructions.get("user_answer") or "").strip()
                if ua:
                    return {
                        "response": ua,
                        "instructions": instructions,
                        "task_results": {},
                        "method": "gemini_direct_answer",
                        "query_type": "analytical"
                    }
            
            # Step 4: Execute data analysis tasks
            if instructions.get("needs_db_query") and instructions.get("db_query"):
                await self._execute_db_query(instructions["db_query"])
            
            task_results = await self._execute_tasks(instructions)
            logger.info(f"⚙️ Task execution completed")
            
            # Step 5: Gemini Summarizer - compose dynamic response
            final_response = await self._gemini_summarizer(user_query, task_results)
            logger.info(f"✅ Gemini final response generated")
            
            return {
                "response": final_response,
                "instructions": instructions,
                "task_results": task_results,
                "method": "analytical_gemini_full",
                "query_type": "analytical"
            }
            
        except Exception as e:
            logger.error(f"❌ Analytical query processing failed: {e}")
            return {
                "response": f"I encountered an issue analyzing the data: {str(e)}. Please try rephrasing your question.",
                "method": "analytical_error",
                "error": str(e)
            }
    
    async def _execute_db_query(self, db_query):
        """Execute database query to pre-filter data if needed."""
        try:
            if isinstance(db_query, dict) and os.getenv("USE_POSTGRES", "0") == "1" and asyncpg is not None:
                df = await self._execute_safe_query(db_query)
                self._query_data_override = df
            else:
                # Fallback: emulate simple bounds on in-memory data
                sql = str(db_query)
                df = self.data.copy()
                import re
                m = re.search(r"latitude\s*between\s*([\-\d\.]+)\s*and\s*([\-\d\.]+)", sql, re.I)
                if m:
                    lo, hi = float(m.group(1)), float(m.group(2))
                    df = df[(df["latitude"] >= lo) & (df["latitude"] <= hi)]
                m = re.search(r"longitude\s*between\s*([\-\d\.]+)\s*and\s*([\-\d\.]+)", sql, re.I)
                if m:
                    lo, hi = float(m.group(1)), float(m.group(2))
                    df = df[(df["longitude"] >= lo) & (df["longitude"] <= hi)]
                self._query_data_override = df
        except Exception as e:
            logger.error(f"❌ DB query execution failed: {e}")
            self._query_data_override = None
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return {
                "response": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "method": "error",
                "error": str(e)
            }
    
    def _validate_instructions(self, user_query: str, instructions: Dict[str, Any]):
        """
        Lightweight validation: ensure shapes and types are sane. Gemini decides tool usage.
        """
        try:
            _ = bool(instructions.get("needs_db_query", False))
            _ = instructions.get("db_query", "")
            _ = list(instructions.get("visualization_requests", []))
            _ = list(instructions.get("analysis_requests", []))
            _ = str(instructions.get("user_answer", ""))
        except Exception as e:
            raise ValueError(f"Invalid instructions shape: {e}")
        logger.info("✅ Validation passed (lightweight)")

    def _generate_explainable_metadata(self, user_query: str, instructions: Dict[str, Any], 
                                     task_results: Dict[str, Any], requires_data: bool) -> Dict[str, Any]:
        """
        Generate metadata explaining how the answer was derived.
        Makes it transparent whether results are computed vs hallucinated.
        """
        tools_executed = []
        data_sources = []
        
        # Check what tools actually ran
        if instructions.get("needs_db_query") and instructions.get("db_query"):
            tools_executed.append("database_query")
            data_sources.append("filtered_database")
        
        if task_results.get("mcp_results"):
            for tool_name in task_results["mcp_results"].keys():
                tools_executed.append(f"analysis_{tool_name}")
                data_sources.append("computed_analysis")
        
        if task_results.get("visualizations"):
            tools_executed.append("visualization_generation")
            data_sources.append("data_visualization")
        
        # Determine answer reliability
        if requires_data and not tools_executed:
            reliability = "UNRELIABLE - No tools executed for data query"
        elif requires_data and tools_executed:
            reliability = "RELIABLE - Based on computed results"
        else:
            reliability = "INFORMATIONAL - General knowledge"
        
        return {
            "query_requires_data": requires_data,
            "tools_executed": tools_executed,
            "data_sources": data_sources,
            "reliability": reliability,
            "answer_based_on": "computed_results" if tools_executed else "cached_stats_only",
            "validation_passed": True  # If we get here, validation passed
            }
    
    async def _gemini_director(self, user_query: str) -> Dict[str, Any]:
        """Enhanced Gemini Director with robust region mapping and validation."""
        try:
            prompt = f"""
            You are an expert oceanographic data analysis director. Analyze this user query and generate precise execution instructions.

            User Query: "{user_query}"

            Available Data Context:
            - ARGO float measurements (temperature, salinity, pressure, depth, location)
            - {len(self.data)} total measurements from Indian Ocean region
            - Geographic coverage: {self.data['latitude'].min():.2f}°S to {self.data['latitude'].max():.2f}°S, {self.data['longitude'].min():.2f}°E to {self.data['longitude'].max():.2f}°E

            CRITICAL REGION MAPPING RULES:
            1. "Indian Ocean" or "indian ocean region" → MUST set region: "indian", lat_range: [-12.37, -8.05], lon_range: [39.72, 55.95]
            2. "equator" or "equatorial" or "near equator" → MUST set region: "equator", lat_range: [-5, 5]
            3. "tropical" or "tropics" → MUST set region: "tropical", lat_range: [-23.5, 23.5]
            4. "pacific ocean" → region: "pacific" (note: will return empty as we only have Indian Ocean data)
            5. "atlantic ocean" → region: "atlantic" (note: will return empty as we only have Indian Ocean data)

            SQL PLANNING RULES:
            - If the query mentions time, trend, over years/decade, averaging by period,
              monthly/seasonal/yearly values, or row limits → MUST set needs_db_query: true
              and return a structured db_query with columns, filters, aggregations, group_by, order_by, limit.
            - Prefer aggregations computed in SQL (e.g., AVG) with GROUP BY periods.
            - Otherwise, only use analysis_requests/visualization_requests when SQL is not needed.

            MANDATORY RULES:
            1. For time/trend/period aggregations → needs_db_query: true with db_query including aggregations and group_by
            2. For ANY region mentioned → include region mapping and/or SQL filters
            3. NEVER fabricate numbers in user_answer; leave empty for analytical queries
            4. Validate region mapping when provided

            EXAMPLE RESPONSES:

            Query: "Calculate the average sea surface temperature in the Indian Ocean and show me the trend over the last decade."
            {{
                "needs_db_query": true,
                "db_query": {{
                    "table": "argo_profiles",
                    "columns": ["date"],
                    "filters": {{ "region": "indian", "date": {{"op":"between","low":"2014-01-01","high":"2024-01-01"}} }},
                    "aggregations": [{{ "op": "AVG", "column": "temperature", "as": "avg_temp_c" }}],
                    "group_by": ["YEAR(date) AS year"],
                    "order_by": [{{"column":"year","direction":"ASC"}}],
                    "limit": 600
                }},
                "analysis_requests": [],
                "visualization_requests": ["plot_time_series"],
                "user_answer": "",
                "data_filters": {{ "region": "indian" }}
            }}

            Query: "What's the temperature near the equator?"
            {{
                "needs_db_query": false,
                "db_query": "",
                "analysis_requests": ["calculate_statistics"],
                "visualization_requests": [],
                "user_answer": "",
                "data_filters": {{
                    "region": "equator",
                    "lat_range": [-5, 5]
                }}
            }}

            Query: "Plot temperature vs depth in tropical region"
            {{
                "needs_db_query": false,
                "db_query": "",
                "analysis_requests": [],
                "visualization_requests": ["plot_depth_profile"],
                "user_answer": "",
                "data_filters": {{
                    "region": "tropical",
                    "lat_range": [-23.5, 23.5]
                }}
            }}

            Respond with ONLY valid JSON following this exact format. No explanations.
            """
            
            response = self._safe_gemini_call(prompt)
            raw = response.text.strip()
            
            # Extract JSON from response
            first = raw.find('{')
            last = raw.rfind('}')
            if first != -1 and last != -1 and last > first:
                raw = raw[first:last+1]

            try:
                plan = json.loads(raw)
                
                # POST-PROCESSING VALIDATION (Critical Fix)
                plan = self._validate_and_fix_plan(plan, user_query)
                
                logger.info(f"� Enhanced Gemini plan: {plan}")
                return plan
                
            except Exception as e:
                logger.error(f"❌ JSON parsing failed: {e}")
                logger.error(f"Raw Gemini response: {raw}")
                return self._emergency_plan_generation(user_query)
                
        except Exception as e:
            logger.error(f"❌ Gemini Director failed: {e}")
            return self._emergency_plan_generation(user_query)

    def _validate_and_fix_plan(self, plan: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Post-process and validate Gemini plan, fixing common issues."""
        query_lower = user_query.lower()
        
        # Ensure basic structure
        if not isinstance(plan, dict):
            logger.warning("🔧 Plan is not a dict, using emergency generation")
            return self._emergency_plan_generation(user_query)
        
        # Initialize missing fields
        plan.setdefault("needs_db_query", False)
        plan.setdefault("db_query", "")
        plan.setdefault("analysis_requests", [])
        plan.setdefault("visualization_requests", [])
        plan.setdefault("user_answer", "")
        plan.setdefault("data_filters", {})
        
        filters = plan["data_filters"]
        
        # CRITICAL FIX: Force region mapping if Gemini missed it
        region_mappings = {
            "indian ocean": {"region": "indian", "lat_range": [-12.37, -8.05], "lon_range": [39.72, 55.95]},
            "equator": {"region": "equator", "lat_range": [-5, 5]},
            "equatorial": {"region": "equator", "lat_range": [-5, 5]},
            "tropical": {"region": "tropical", "lat_range": [-23.5, 23.5]},
            "tropics": {"region": "tropical", "lat_range": [-23.5, 23.5]},
        }
        
        for region_phrase, region_config in region_mappings.items():
            if region_phrase in query_lower:
                if not filters.get("region") or filters.get("region") != region_config["region"]:
                    filters.update(region_config)
                    logger.info(f"🔧 Auto-fixed region mapping: {region_phrase} → {region_config}")
                break
        
        # Do not auto-inject analysis tools or override DB usage; let Gemini decide entirely
        
        logger.info(f"✅ Plan validation complete: {plan}")
        return plan

    def _emergency_plan_generation(self, user_query: str) -> Dict[str, Any]:
        """Emergency fallback with aggressive data finding."""
        query_lower = user_query.lower()
        logger.warning(f"🚨 Emergency plan generation for: {user_query}")
        
        # Default plan structure
        plan = {
            "needs_db_query": False,
            "db_query": "",
            "analysis_requests": [],
            "visualization_requests": [],
            "user_answer": "",
            "data_filters": {}
        }
        
        # Emergency region mapping
        if "indian ocean" in query_lower:
            plan["data_filters"] = {
                "region": "indian",
                "lat_range": [-12.37, -8.05],
                "lon_range": [39.72, 55.95]
            }
        elif "equator" in query_lower:
            plan["data_filters"] = {
                "region": "equator", 
                "lat_range": [-5, 5]
            }
        elif "tropical" in query_lower:
            plan["data_filters"] = {
                "region": "tropical",
                "lat_range": [-23.5, 23.5]
            }
        
        # Do not inject default analyses/visualizations; return empty plan for Gemini to clarify
        
        logger.info(f"🚨 Emergency plan: {plan}")
        return plan
    
    def _fallback_plan_generation(self, user_query: str) -> Dict[str, Any]:
        """Generate intelligent fallback plan when Gemini director fails."""
        query_lower = user_query.lower()
        
    def _fallback_plan_generation(self, user_query: str) -> Dict[str, Any]:
        """Enhanced fallback plan with proper region mapping"""
        query_lower = user_query.lower()
        
        # Initialize default plan
        plan = {
            "needs_db_query": False,  # Use CSV data directly
            "db_query": "",
            "analysis_requests": [],
            "visualization_requests": [],
            "user_answer": "",
            "data_filters": {}
        }
        
        # Region mapping - CRITICAL FIX
        if "indian ocean" in query_lower or "indian" in query_lower:
            plan["data_filters"]["region"] = "indian"
            plan["data_filters"]["lat_range"] = [-12.37, -8.05]
            plan["data_filters"]["lon_range"] = [39.72, 55.95]
            logger.info("🔧 Fallback: Applied Indian Ocean region mapping")
        elif "equator" in query_lower:
            plan["data_filters"]["region"] = "equator"
            plan["data_filters"]["lat_range"] = [-5, 5]
            logger.info("🔧 Fallback: Applied equatorial region mapping")
        elif "tropical" in query_lower:
            plan["data_filters"]["region"] = "tropical"
            plan["data_filters"]["lat_range"] = [-23.5, 23.5]
            logger.info("🔧 Fallback: Applied tropical region mapping")
        
        # Analysis requests based on keywords
        if any(word in query_lower for word in ['average', 'mean', 'statistics']):
            plan["analysis_requests"].append('calculate_statistics')
            logger.info("🔧 Fallback: Added statistics analysis")
        elif any(word in query_lower for word in ['highest', 'maximum', 'max']):
            if 'temperature' in query_lower:
                plan["analysis_requests"].append('max_temperature')
            elif 'salinity' in query_lower:
                plan["analysis_requests"].append('max_salinity')
            else:
                plan["analysis_requests"].append('calculate_statistics')
        elif any(word in query_lower for word in ['count', 'many', 'number', 'total']):
            plan["analysis_requests"].append('calculate_statistics')
        else:
            # Default for analytical queries
            plan["analysis_requests"].append('calculate_statistics')
        
        # Visualization requests
        if any(word in query_lower for word in ['plot', 'map', 'chart', 'graph', 'visualize', 'show']):
            if 'temperature' in query_lower and 'map' in query_lower:
                plan["visualization_requests"].append('plot_temperature_map')
            elif 'temperature' in query_lower and 'profile' in query_lower:
                plan["visualization_requests"].append('plot_depth_profile')
            elif 'salinity' in query_lower and 'profile' in query_lower:
                plan["visualization_requests"].append('plot_salinity_profile')
        
        logger.info(f"📋 Fallback plan generated: {plan}")
        return plan
    
    async def _execute_tasks(self, instructions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG and MCP tasks based on AI instructions"""
        results = {
            "rag_context": {},
            "mcp_results": {},
            "visualizations": {},
            "data_summary": {}
        }
        
        def _normalize_viz_request(v: Any) -> Dict[str, Any]:
            """Map free-text viz requests to known tool types and attach filters."""
            default = {"type": "", "filters": instructions.get("data_filters", {})}
            if isinstance(v, dict):
                t = v.get("type") or v.get("tool") or ""
                f = v.get("filters", instructions.get("data_filters", {}))
                return {"type": str(t), "filters": f}
            text = str(v).lower()
            if "temperature vs salinity" in text or ("scatter" in text and "salinity" in text) or "plot_temperature_vs_salinity" in text:
                default["type"] = "plot_temperature_vs_salinity"
                return default
            if ("salinity" in text and "depth" in text) or "plot_salinity_profile" in text:
                default["type"] = "plot_salinity_profile"
                return default
            if "depth profile" in text or "plot_depth_profile" in text:
                default["type"] = "plot_depth_profile"
                return default
            if (("temperature" in text and ("map" in text or "distribution" in text)) or "plot_temperature_map" in text):
                default["type"] = "plot_temperature_map"
                return default
            # Fallback to pass-through (will be handled by unknown mapping)
            default["type"] = text.strip()
            return default

        def _normalize_analysis_request(a: Any) -> Dict[str, Any]:
            default = {"type": "", "filters": instructions.get("data_filters", {})}
            if isinstance(a, dict):
                t = a.get("type") or a.get("analysis") or a.get("tool") or ""
                f = a.get("filters", instructions.get("data_filters", {}))
                return {"type": str(t), "filters": f}
            text = str(a).lower()
            if "highest" in text and "temperature" in text:
                default["type"] = "max_temperature"
                return default
            if ("max" in text or "highest" in text) and "salinity" in text:
                default["type"] = "max_salinity"
                return default
            if "thermocline" in text:
                default["type"] = "thermocline_analysis"
                return default
            if "statistic" in text or "summary" in text:
                default["type"] = "statistics_summary"
                return default
            if "filter" in text and ("spatial" in text or "region" in text):
                default["type"] = "filter_spatial_data"
                return default
            default["type"] = text.strip()
            return default
        
        # Execute RAG queries
        for query in instructions.get("rag_queries", []):
            if query in self.rag_context:
                results["rag_context"][query] = self.rag_context[query]
        
        # Execute MCP tasks
        for task in instructions.get("mcp_tasks", []):
            try:
                # Handle both string tasks and dict tasks
                if isinstance(task, str):
                    # Extract task name from description (e.g., "filter_spatial_data: Apply geographic bounds..." -> "filter_spatial_data")
                    if ':' in task:
                        task_name = task.split(':')[0].strip()
                    else:
                        task_name = task
                    
                    # Remove numbered prefixes (e.g., "1. filter_spatial_data" -> "filter_spatial_data")
                    if '.' in task_name and task_name[0].isdigit():
                        task_name = task_name.split('.', 1)[1].strip()
                    
                    task_params = instructions.get("data_filters", {})
                elif isinstance(task, dict):
                    # Extract task name and parameters from dict format
                    if 'task' in task:
                        task_name = task['task']
                        task_params = task.get('parameters', instructions.get("data_filters", {}))
                    elif 'tool' in task:
                        task_name = task['tool']
                        task_params = task.get('parameters', instructions.get("data_filters", {}))
                    else:
                        # Try to get the first key as task name
                        task_name = list(task.keys())[0]
                        task_params = task.get(task_name, instructions.get("data_filters", {}))
                else:
                    logger.error(f"❌ Unknown task format: {task}")
                    continue
                
                if task_name in self.mcp_tools:
                    result = await asyncio.wait_for(self.mcp_tools[task_name](task_params), timeout=self._tool_timeout)
                    results["mcp_results"][task_name] = result
                else:
                    logger.error(f"❌ Unknown MCP task: {task_name}")
                    results["mcp_results"][task_name] = {"error": f"Unknown task: {task_name}"}
            except Exception as e:
                logger.error(f"❌ MCP task {task} failed: {e}")
                results["mcp_results"][str(task)] = {"error": str(e)}
        
        # Execute visualizations using MCP server tools prioritized
        for viz_request in instructions.get("visualization_requests", []):
            try:
                # normalize time-series requests to our generic line plot
                if isinstance(viz_request, dict):
                    vtype = viz_request.get('type') or viz_request.get('tool') or ''
                else:
                    vtype = str(viz_request)
                if vtype in ['plot_time_series','time_series','generate_time_series','plot_timeseries']:
                    viz_request = 'plot_line'
                    # set default axes for time-series if not provided
                    if 'data_filters' in instructions:
                        instructions['data_filters'].setdefault('x_axis','year')
                        # default Y for avg temperature aggregation alias
                        instructions['data_filters'].setdefault('y_axis','avg_temp_c')
                viz_name = viz_request if isinstance(viz_request, str) else viz_request.get('type', 'unknown')
                logger.info(f"🎨 Generating MCP visualization: {viz_name}")
                
                # Force use of MCP server visualization tools
                if self.mcp_server:
                    # Map requests to comprehensive MCP server tools (only available ones)
                    mcp_tool_mapping = {
                        'plot_temperature_profile': 'generate_temperature_profile_plot',
                        'temperature_profile': 'generate_temperature_profile_plot', 
                        'plot_depth_profile': 'generate_temperature_profile_plot',
                        'depth_profile': 'generate_temperature_profile_plot',
                        
                        'plot_geographic_map': 'generate_geographic_map',
                        'geographic_map': 'generate_geographic_map',
                        'plot_temperature_map': 'generate_geographic_map',
                        
                        'time_series': 'generate_time_series_plot',
                        'correlation_matrix': 'generate_correlation_matrix',
                        
                        # For T-S diagrams, use fallback to basic tools since not available in MCP server
                        'plot_ts_diagram': 'temperature_salinity_scatter',
                        'ts_diagram': 'temperature_salinity_scatter',
                        'temperature_salinity_scatter': 'temperature_salinity_scatter',
                        
                        # For salinity profiles, use fallback to basic tools
                        'plot_salinity_profile': 'plot_salinity_profile',
                        'salinity_profile': 'plot_salinity_profile'
                    }
                    
                    # Check if we have a direct mapping to MCP server tool
                    mcp_tool_name = mcp_tool_mapping.get(viz_name, viz_name)
                    
                    if mcp_tool_name in self.mcp_tools:
                        logger.info(f"🚀 Using MCP server tool: {mcp_tool_name}")
                        # Apply filters to get working dataset
                        filtered_data = self._apply_filters(self.data, instructions.get('data_filters', {}))
                        result = await self.mcp_tools[mcp_tool_name](filtered_data)
                        
                        if result and result.get('ok'):
                            results["visualizations"][viz_name] = result
                            logger.info(f"✅ MCP Visualization {viz_name} generated successfully")
                        else:
                            error_msg = result.get('error', 'No result returned') if result else 'Tool returned None'
                            logger.warning(f"⚠️ MCP Visualization {viz_name} failed: {error_msg}")
                            results["visualizations"][viz_name] = {
                                'ok': False,
                                'error': error_msg,
                                'attempted_tool': mcp_tool_name
                            }
                    else:
                        logger.warning(f"⚠️ MCP tool {mcp_tool_name} not available, trying fallback")
                        # Only use basic visualization as last resort
                        viz = await asyncio.wait_for(self._generate_visualization(viz_name, instructions.get("data_filters", {})), timeout=self._tool_timeout)
                        results["visualizations"][viz_name] = viz
                        logger.info(f"✅ Fallback visualization generated: {viz_name}")
                else:
                    logger.warning("⚠️ MCP server not available, using basic visualization")
                    # Use basic visualization method
                    viz = await asyncio.wait_for(self._generate_visualization(viz_name, instructions.get("data_filters", {})), timeout=self._tool_timeout)
                    results["visualizations"][viz_name] = viz
                    logger.info(f"✅ Basic visualization generated: {viz_name}")
            
            except Exception as e:
                logger.error(f"❌ Visualization {viz_request} failed: {e}")
                results["visualizations"][str(viz_request)] = {"error": str(e)}

        # Execute analyses using MCP server tools prioritized
        analysis_requests = instructions.get("analysis_requests", [])
        if analysis_requests:
            logger.info(f"🔬 Executing {len(analysis_requests)} analysis tasks (MCP only; no dict fallbacks)")
            
            for analysis_request in analysis_requests:
                task_name = analysis_request if isinstance(analysis_request, str) else analysis_request.get('type', 'unknown')
                logger.info(f"⚙️ Running MCP analysis: {task_name}")
                
                try:
                    # Get data filters (NOT filtered data!)
                    data_filters = instructions.get('data_filters', {})
                    logger.info(f"📋 Using filters for {task_name}: {data_filters}")
                    
                    # Execute with filters (let the tool do its own filtering)
                    if task_name in self.mcp_tools:
                        tool_func = self.mcp_tools[task_name]
                        
                        # Pass filters to the tool function (not pre-filtered data)
                        if asyncio.iscoroutinefunction(tool_func):
                            result = await tool_func(data_filters)
                        else:
                            result = tool_func(data_filters)
                        
                        if result and result.get('ok'):
                            results['mcp_results'][task_name] = result
                            logger.info(f"✅ {task_name} completed successfully")
                        else:
                            error_msg = result.get('error', 'No valid result returned') if result else 'Tool returned None'
                            logger.warning(f"⚠️ {task_name} failed: {error_msg}")
                            results['mcp_results'][task_name] = result or {'ok': False, 'error': 'No result returned'}
                    else:
                        logger.warning(f"⚠️ Unknown analysis task (skipped): {task_name}")
                        results['mcp_results'][task_name] = {'ok': False, 'error': f'Unknown task: {task_name}'}
                
                except Exception as e:
                    logger.error(f"❌ Analysis {task_name} failed: {e}")
                    results['mcp_results'][task_name] = {'ok': False, 'error': str(e)}
        
            # Do not synthesize summaries when MCP failed; only include if explicitly requested
            results["data_summary"] = {}
        
        return results
    
    async def _gemini_summarizer(self, user_query: str, task_results: Dict[str, Any]) -> str:
        """Summarize results.
        If USE_LLM_PROSE_SUMMARY=1, produce short LLM prose using ONLY computed fields.
        Otherwise, return deterministic numeric summary.
        """
        try:
            if os.getenv("USE_LLM_PROSE_SUMMARY", "0") == "1":
                # Prepare minimal, JSON-safe facts only
                mcp = task_results.get("mcp_results", {}) or {}
                safe = {
                    "max_temperature": mcp.get("max_temperature"),
                    "max_salinity": mcp.get("max_salinity"),
                    "statistics": (mcp.get("calculate_statistics") or mcp.get("statistics_summary"))
                }
                safe = self._make_json_safe(safe)
                # If nothing to summarize, fallback to deterministic guidance
                if not any(bool(v) for v in safe.values()):
                    return (
                        "🤝 I'm here to help analyze the data. Try: 'calculate statistics', 'highest temperature', "
                        "or 'temperature vs salinity scatter'."
                    )
                prompt = (
                    "You are a summarizer. Write a brief, factual summary (<= 4 short sentences) "
                    "using ONLY the fields provided. DO NOT invent values, tables, or context. "
                    "Prefer extractive phrasing with explicit numbers. If a field is missing, omit it.\n\n"
                    f"User Query: {user_query}\n"
                    f"Computed Results (JSON):\n{json.dumps(safe, indent=2)}\n\n"
                    "Output only the summary text."
                )
                try:
                    if getattr(self, "_use_gemini_director", False) and getattr(self, "_gemini_model", None) is not None:
                        resp = self._safe_gemini_call(prompt)
                        return (resp.text or "").strip() or "(no content)"
                    else:
                        resp = self.llm.generate_content(prompt)
                        return (resp.text or "").strip() or "(no content)"
                except Exception as e:
                    logger.warning(f"LLM prose summary failed, falling back: {e}")
                    # Fall through to deterministic summary below
            
            mcp = task_results.get("mcp_results", {})

            logger.info(f"🔍 MCP Results keys: {list(mcp.keys())}")
            for key, value in mcp.items():
                logger.info(f"🔍 {key}: {type(value)} - {value}")

            # Check for insufficient data signals
            for v in list(mcp.values()) + list(task_results.get("visualizations", {}).values()):
                if isinstance(v, dict) and v.get("insufficient_data"):
                    reason = v.get("reason", "insufficient data")
                    return f"⚠️ Unable to compute: {reason}. Try broadening filters or choosing the Indian Ocean dataset."

            # Get statistics from either canonical or alias key
            stats_result = mcp.get("calculate_statistics") or mcp.get("statistics_summary") or {}
            stats = stats_result.get("statistics", {}) if isinstance(stats_result, dict) else {}
            
            # Get max temperature and salinity results - check multiple possible key variations
            max_t = (mcp.get("max_temperature", {}) or 
                    mcp.get("max_temperature_analysis", {}) or
                    mcp.get("temperature_max", {}))
            
            max_s = (mcp.get("max_salinity", {}) or
                    mcp.get("max_salinity_analysis", {}) or
                    mcp.get("salinity_max", {}))

            parts = []
            
            # Handle max temperature
            if max_t and not max_t.get("error"):
                temp_val = max_t.get('temperature_c', 'N/A')
                float_id = max_t.get('float_id', 'Unknown')
                profile_idx = max_t.get('profile_index', 'Unknown')
                if temp_val != 'N/A':
                    parts.append(f"🌡️ Highest temperature: {temp_val:.2f}°C (Float {float_id}, Profile {profile_idx})")
            
            # Handle max salinity  
            if max_s and not max_s.get("error"):
                sal_val = max_s.get('salinity_psu', 'N/A')
                float_id = max_s.get('float_id', 'Unknown')
                profile_idx = max_s.get('profile_index', 'Unknown')
                if sal_val != 'N/A':
                    parts.append(f"🧂 Highest salinity: {sal_val:.3f} PSU (Float {float_id}, Profile {profile_idx})")
            
            # Handle statistics
            if stats:
                tstats = stats.get("temperature", {})
                sstats = stats.get("salinity", {})
                if tstats and sstats:
                    # If user asked for averages/means, prioritize those
                    ql = (user_query or "").lower()
                    want_avg = any(k in ql for k in ["average", "mean", "avg"])
                    if want_avg:
                        tmean = tstats.get('mean')
                        smean = sstats.get('mean')
                        if tmean is not None:
                            parts.append(f"📈 Average temperature: {tmean:.2f}°C")
                        if smean is not None:
                            parts.append(f"📈 Average salinity: {smean:.3f} PSU")
                    else:
                        temp_min = tstats.get('min')
                        temp_max = tstats.get('max')
                        sal_min = sstats.get('min')
                        sal_max = sstats.get('max')
                        if None not in (temp_min, temp_max, sal_min, sal_max):
                            parts.append(f"📊 Temperature: {temp_min:.2f}–{temp_max:.2f}°C | Salinity: {sal_min:.3f}–{sal_max:.3f} PSU")

            if not parts:
                # If no computed outputs, fall back to Gemini knowledge-based response for ANY query
                logger.info("🧠 No computed results; returning Gemini knowledge-based response")
                return await self._knowledge_based_response(user_query, f"MCP Results: {list(mcp.keys())}")
            
            return " | ".join(parts)
            
        except Exception as e:
            logger.error(f"❌ Deterministic summarizer failed: {e}")
            return f"❌ Results computed but could not format summary: {str(e)}"

    async def _knowledge_based_response(self, user_query: str, context_text: str) -> str:
        """Generate comprehensive knowledge-based responses using Gemini's expertise when data is limited"""
        try:
            # Enhanced prompt for knowledge-based responses
            prompt = f"""As an expert oceanographer and marine scientist, provide a comprehensive, educational response to this user's query. While the current ARGO dataset may have limitations, use your extensive knowledge of oceanography, marine biology, and climate science to provide valuable insights.

User Query: {user_query}

Current Data Context: {context_text}

Please provide a detailed response that:

1. **Acknowledges the data context**: Briefly mention what data is available in the current ARGO dataset
2. **Provides comprehensive scientific knowledge**: Draw from your expertise to explain the scientific concepts related to the query
3. **Includes real-world examples**: Provide specific examples from known oceanographic phenomena
4. **Explains mechanisms and processes**: Describe the underlying scientific processes involved
5. **Discusses regional variations**: Explain how different ocean regions might exhibit different patterns
6. **Suggests methodologies**: Describe what data and methods scientists typically use to study these phenomena
7. **Educational context**: Make the response informative for someone learning about oceanography
8. **Future directions**: Suggest related topics or advanced analyses that would be valuable

Structure your response in a clear, engaging way that combines scientific accuracy with accessibility. Provide the kind of comprehensive answer that a marine science professor would give to an interested student.

Even if the specific analysis cannot be performed with current data, make this a valuable learning experience about the topic."""

            response = self._safe_gemini_call(prompt)
            
            if response and hasattr(response, 'text'):
                knowledge_response = response.text.strip()
                if knowledge_response and len(knowledge_response) > 100:
                    return knowledge_response
            
            # Fallback if Gemini fails
            return self._basic_knowledge_fallback(user_query)
            
        except Exception as e:
            logger.error(f"Knowledge-based response failed: {e}")
            return self._basic_knowledge_fallback(user_query)

    def _basic_knowledge_fallback(self, user_query: str) -> str:
        """Basic fallback response when all else fails"""
        if "phytoplankton" in user_query.lower() or "bloom" in user_query.lower():
            return """🌊 **Phytoplankton and Oceanographic Conditions**

While the current ARGO dataset focuses on physical oceanographic measurements (temperature, salinity, pressure), phytoplankton blooms are indeed strongly influenced by these parameters. Here's what oceanographers know about these relationships:

**Key Relationships:**
- **Salinity gradients** often indicate freshwater inputs that bring nutrients, potentially triggering blooms
- **Temperature fronts** create mixing zones where nutrients are brought to surface waters
- **Upwelling regions** (detectable through temperature/salinity patterns) are major bloom zones

**Data Requirements for Bloom Analysis:**
- BGC-ARGO floats with chlorophyll sensors
- Satellite ocean color data
- Nutrient measurements
- Light availability data

The ARGO temperature and salinity data you have can identify oceanographic features that *correlate* with bloom conditions, even without direct biological measurements."""
        
        return "🤝 I'm here to help analyze oceanographic data. The current dataset contains ARGO temperature, salinity, and pressure measurements. Try asking about temperature patterns, salinity distributions, or water mass characteristics."
    
    # MCP Tool Implementations
    async def _plot_temperature_map(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Create temperature distribution map"""
        try:
            # Filter data if needed
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            data = self._apply_filters(data, filters)
            if data.empty:
                return {"type": "temperature_map", "ok": False, "insufficient_data": True, "reason": "No data for selected region/filters.", "filters": self._make_json_safe(filters)}
            
            # Create temperature map
            fig = go.Figure()
            fig.add_trace(go.Scattergeo(
                lat=data['latitude'],
                lon=data['longitude'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=data['temperature'],
                    colorscale='Viridis',
                    colorbar=dict(title="Temp (°C)"),
                    opacity=0.8
                ),
                text=[f"Temp: {temp:.1f}°C<br>Sal: {sal:.2f}" for temp, sal in zip(data['temperature'], data['salinity'])],
                hovertemplate='%{text}<extra></extra>',
                name='ARGO Floats'
            ))
            fig.update_geos(
                projection_type="natural earth",
                showcountries=True,
                showsubunits=True,
            )
            fig.update_layout(
                margin=dict(r=0, t=0, l=0, b=0),
                height=420,
                template='plotly_dark'
            )
            
            return {
                "type": "temperature_map",
                "description": "Temperature distribution across ARGO float locations",
                "figure": fig.to_json(),
                "data_points": len(data),
                "ok": True
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    async def _plot_salinity_profile(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Create salinity profile"""
        try:
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            data = self._apply_filters(data, filters)
            if data.empty:
                return {"type": "salinity_profile", "ok": False, "insufficient_data": True, "reason": "No data for selected region/filters.", "filters": self._make_json_safe(filters)}
            
            # Create salinity profile
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['salinity'],
                y=data['pressure'],
                mode='markers',
                marker=dict(size=4, color='#00CED1'),
                name='Salinity Profile'
            ))
            
            fig.update_layout(
                title="Salinity vs Pressure Profile",
                xaxis_title="Salinity (PSU)",
                yaxis_title="Pressure (dbar)",
                yaxis=dict(autorange='reversed'),
                template='plotly_dark',
                height=400
            )
            
            return {
                "type": "salinity_profile",
                "description": "Salinity profile showing salinity vs pressure",
                "figure": fig.to_json(),
                "data_points": len(data),
                "ok": True
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def _analyze_thermocline(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thermocline structure"""
        try:
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            data = self._apply_filters(data, filters)
            if data.empty:
                return {"type": "thermocline_analysis", "ok": False, "insufficient_data": True, "reason": "No data for selected region/filters.", "filters": self._make_json_safe(filters)}
            
            # Simple thermocline analysis
            temp_gradient = data['temperature'].diff().abs().mean()
            
            return {
                "type": "thermocline_analysis",
                "description": "Thermocline structure analysis",
                "temperature_gradient": float(temp_gradient),
                "data_points": len(data),
                "ok": True
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def _calculate_statistics(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical summaries with progressive filtering."""
        try:
            # Start with full dataset
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            
            logger.info(f"📊 Calculating statistics with filters: {filters}")
            
            # Use progressive filtering to ensure we get data
            filtered_data = self._apply_filters_with_fallback(data, filters)
            
            if filtered_data.empty:
                logger.error("❌ No data available after all filtering strategies")
                return {
                    "type": "statistics", 
                    "ok": False, 
                    "insufficient_data": True, 
                    "reason": "No data found even with relaxed filters", 
                    "filters": self._make_json_safe(filters),
                    "original_count": len(data)
                }
            
            # Calculate comprehensive statistics
            stats = {
                "total_measurements": len(filtered_data),
                "unique_floats": filtered_data['profile_index'].nunique(),
                "temperature": {
                    "mean": float(filtered_data['temperature'].mean()),
                    "std": float(filtered_data['temperature'].std()),
                    "min": float(filtered_data['temperature'].min()),
                    "max": float(filtered_data['temperature'].max())
                },
                "salinity": {
                    "mean": float(filtered_data['salinity'].mean()),
                    "std": float(filtered_data['salinity'].std()),
                    "min": float(filtered_data['salinity'].min()),
                    "max": float(filtered_data['salinity'].max())
                },
                "pressure": {
                    "mean": float(filtered_data['pressure'].mean()),
                    "std": float(filtered_data['pressure'].std()),
                    "min": float(filtered_data['pressure'].min()),
                    "max": float(filtered_data['pressure'].max())
                }
            }
            
            # Add geographic context if region specified
            region_info = {}
            if filters.get("region"):
                region_info["region"] = filters["region"]
                if "latitude" in filtered_data.columns:
                    region_info["latitude_range"] = [
                        float(filtered_data['latitude'].min()),
                        float(filtered_data['latitude'].max())
                    ]
                if "longitude" in filtered_data.columns:
                    region_info["longitude_range"] = [
                        float(filtered_data['longitude'].min()),
                        float(filtered_data['longitude'].max())
                    ]
            
            result = {
                "type": "statistics",
                "description": f"Statistical analysis for {filters.get('region', 'selected')} region",
                "statistics": stats,
                "region_info": region_info,
                "data_points_used": len(filtered_data),
                "original_data_points": len(data),
                "filtering_success_rate": round(len(filtered_data) / len(data) * 100, 1),
                "ok": True
            }
            
            logger.info(f"✅ Statistics calculated: {len(filtered_data)} records from {filters.get('region', 'unknown')} region")
            return result
            
        except Exception as e:
            logger.error(f"❌ Statistics calculation failed: {e}")
            return {"ok": False, "error": str(e), "type": "statistics"}

    def _filter_spatial_data(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Filter data by spatial criteria"""
        try:
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            filtered = self._apply_filters(data, filters)
            if filtered.empty:
                return {"type": "spatial_filter", "ok": False, "insufficient_data": True, "reason": "No data for selected region/filters.", "filters": self._make_json_safe(filters)}
            return {
                "type": "spatial_filter",
                "description": "Spatial data filtering applied",
                "filtered_measurements": len(filtered),
                "original_measurements": len(self.data),
                "region": filters.get("ocean", "selected") or "selected",
                "filters": self._make_json_safe(filters),
                "ok": True
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    async def _plot_depth_profile(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Create depth profile visualization"""
        try:
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            data = self._apply_filters(data, filters)
            if data.empty:
                return {"type": "depth_profile", "ok": False, "insufficient_data": True, "reason": "No data for selected region/filters.", "filters": self._make_json_safe(filters)}
            
            # Create depth profile
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['temperature'],
                y=data['pressure'],
                mode='markers',
                marker=dict(size=4, color='#00BFA6'),
                name='Depth Profile'
            ))
            
            fig.update_layout(
                title="Temperature vs Pressure Profile",
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (dbar)",
                yaxis=dict(autorange='reversed'),
                template='plotly_dark',
                height=400
            )
            
            return {
                "type": "depth_profile",
                "description": "Depth profile showing temperature vs pressure",
                "figure": fig.to_json(),
                "data_points": len(data),
                "ok": True
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _plot_temp_sal_scatter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Temperature vs Salinity scatter plot"""
        try:
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            data = self._apply_filters(data, filters)
            if data.empty:
                return {"type": "temperature_salinity_scatter", "ok": False, "insufficient_data": True, "reason": "No data for selected region/filters.", "filters": self._make_json_safe(filters)}
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["salinity"],
                y=data["temperature"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=data.get("depth", data.get("pressure", 0)),
                    colorscale="Viridis",
                    opacity=0.7
                ),
                name="Temp vs Salinity"
            ))
            fig.update_layout(
                title="Temperature vs Salinity",
                xaxis_title="Salinity (PSU)",
                yaxis_title="Temperature (°C)",
                template="plotly_dark",
                height=420
            )
            return {
                "type": "temperature_salinity_scatter",
                "description": "Scatter of temperature vs salinity, colored by depth/pressure",
                "figure": fig.to_json(),
                "data_points": len(data),
                "ok": True
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def _compute_max_temperature(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the float/profile with highest recorded temperature from actual data."""
        try:
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            data = self._apply_filters(data, filters)
            if data.empty:
                return {"type": "max_temperature", "ok": False, "insufficient_data": True, "reason": "No data for selected region/filters.", "filters": self._make_json_safe(filters)}
            if data.empty or "temperature" not in data.columns:
                return {"error": "No temperature data available"}
            idx = data["temperature"].idxmax()
            row = data.loc[idx]
            return {
                "type": "max_temperature",
                "float_id": int(row.get("float_id")) if "float_id" in row and pd.notna(row.get("float_id")) else (int(row.get("profile_index")) if pd.notna(row.get("profile_index")) else None),
                "profile_index": int(row.get("profile_index")) if pd.notna(row.get("profile_index")) else None,
                "temperature_c": float(row["temperature"]),
                "salinity_psu": float(row.get("salinity", float("nan"))),
                "latitude": float(row.get("latitude", float("nan"))),
                "longitude": float(row.get("longitude", float("nan"))),
                "pressure_dbar": float(row.get("pressure", float("nan"))),
                "ok": True
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _compute_max_salinity(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the float/profile with highest recorded salinity from actual data."""
        try:
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            data = self._apply_filters(data, filters)
            if data.empty:
                return {"type": "max_salinity", "ok": False, "insufficient_data": True, "reason": "No data for selected region/filters.", "filters": self._make_json_safe(filters)}
            if data.empty or "salinity" not in data.columns:
                return {"error": "No salinity data available"}
            idx = data["salinity"].idxmax()
            row = data.loc[idx]
            return {
                "type": "max_salinity",
                "float_id": int(row.get("float_id")) if "float_id" in row and pd.notna(row.get("float_id")) else (int(row.get("profile_index")) if pd.notna(row.get("profile_index")) else None),
                "profile_index": int(row.get("profile_index")) if pd.notna(row.get("profile_index")) else None,
                "temperature_c": float(row.get("temperature", float("nan"))),
                "salinity_psu": float(row["salinity"]),
                "latitude": float(row.get("latitude", float("nan"))),
                "longitude": float(row.get("longitude", float("nan"))),
                "pressure_dbar": float(row.get("pressure", float("nan"))),
                "ok": True
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    async def _generate_visualization(self, viz_request: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization based on request"""
        try:
            # Cache lookup
            cache_key = self._viz_cache_key(viz_request, filters)
            cached = self._viz_cache_get(cache_key)
            if cached is not None:
                return cached

            # Map visualization requests to appropriate functions
            viz_map = {
                "temperature_distribution": self._plot_temperature_map,
                "plot_temperature_map": self._plot_temperature_map,  # synonym
                "salinity_profile": self._plot_salinity_profile,
                "plot_salinity_profile": self._plot_salinity_profile,  # synonym
                "depth_profile": self._plot_depth_profile,
                "plot_depth_profile": self._plot_depth_profile,  # synonym
                "temperature_salinity_scatter": self._plot_temp_sal_scatter,
                "plot_temperature_vs_salinity": self._plot_temp_sal_scatter,  # synonym
                "plot_temp_sal_scatter": self._plot_temp_sal_scatter,  # additional synonym
            }
            if viz_request in viz_map:
                result = await viz_map[viz_request](filters)
                # Cache store if successful
                if isinstance(result, dict) and not result.get("error"):
                    self._viz_cache_put(cache_key, result)
                return result
            else:
                return {"error": f"Unknown visualization request: {viz_request}"}
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_data_summary(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data summary"""
        try:
            data = self.data.copy()
            
            return {
                "total_measurements": len(data),
                "unique_floats": data['profile_index'].nunique(),
                "geographic_coverage": {
                        "latitude_range": [float(data['latitude'].min()), float(data['latitude'].max())],
                        "longitude_range": [float(data['longitude'].min()), float(data['longitude'].max())]
                },
                "parameter_ranges": {
                        "temperature": [float(data['temperature'].min()), float(data['temperature'].max())],
                        "salinity": [float(data['salinity'].min()), float(data['salinity'].max())],
                        "pressure": [float(data['pressure'].min()), float(data['pressure'].max())]
                    }
                }
        except Exception as e:
            return {"error": str(e)}

    # --- Generic Plotting Tools ---
    async def _plot_generic_scatter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a generic scatter plot with user-specified x and y axes"""
        try:
            # Extract x and y axis parameters
            x_axis = filters.get('x_axis', filters.get('x', 'longitude'))
            y_axis = filters.get('y_axis', filters.get('y', 'latitude'))
            
            # Filter data
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            data = self._apply_filters(data, filters)
            
            if data.empty:
                return {"type": "scatter", "ok": False, "insufficient_data": True, 
                       "reason": f"No data for selected filters.", "filters": self._make_json_safe(filters)}
            
            # Check if columns exist
            if x_axis not in data.columns:
                return {"type": "scatter", "ok": False, "error": f"Column '{x_axis}' not found"}
            if y_axis not in data.columns:
                return {"type": "scatter", "ok": False, "error": f"Column '{y_axis}' not found"}
            
            # Sample data for performance
            if len(data) > 2000:
                data = data.sample(2000)
            
            # Create scatter plot
            fig = go.Figure()
            
            # Color by depth if available and not one of the axes
            color_col = None
            if 'depth' in data.columns and x_axis != 'depth' and y_axis != 'depth':
                color_col = 'depth'
            elif 'pressure' in data.columns and x_axis != 'pressure' and y_axis != 'pressure':
                color_col = 'pressure'
            
            if color_col:
                fig.add_trace(go.Scatter(
                    x=data[x_axis],
                    y=data[y_axis],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=data[color_col],
                        colorscale='Viridis',
                        colorbar=dict(title=color_col.title()),
                        opacity=0.7
                    ),
                    text=[f"{x_axis}: {x}<br>{y_axis}: {y}<br>{color_col}: {c}" 
                          for x, y, c in zip(data[x_axis], data[y_axis], data[color_col])],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'{x_axis} vs {y_axis}'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=data[x_axis],
                    y=data[y_axis],
                    mode='markers',
                    marker=dict(size=6, color='#FF6B6B', opacity=0.7),
                    text=[f"{x_axis}: {x}<br>{y_axis}: {y}" 
                          for x, y in zip(data[x_axis], data[y_axis])],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'{x_axis} vs {y_axis}'
                ))
            
            fig.update_layout(
                title=f"{x_axis.title()} vs {y_axis.title()}",
                xaxis_title=x_axis.title(),
                yaxis_title=y_axis.title(),
                template='plotly_dark',
                height=500,
                showlegend=False
            )
            
            return {
                "type": "scatter",
                "description": f"Scatter plot of {x_axis} vs {y_axis}",
                "figure": fig.to_json(),
                "data_points": len(data),
                "x_axis": x_axis,
                "y_axis": y_axis,
                "ok": True
            }
            
        except Exception as e:
            logger.error(f"❌ Generic scatter plot failed: {e}")
            return {"type": "scatter", "ok": False, "error": str(e)}

    async def _plot_generic_line(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a generic line plot with user-specified x and y axes"""
        try:
            # Extract x and y axis parameters
            x_axis = filters.get('x_axis', filters.get('x', 'depth'))
            y_axis = filters.get('y_axis', filters.get('y', 'temperature'))
            
            # Filter data
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            data = self._apply_filters(data, filters)
            
            if data.empty:
                return {"type": "line", "ok": False, "insufficient_data": True, 
                       "reason": f"No data for selected filters.", "filters": self._make_json_safe(filters)}
            
            # Check if columns exist
            if x_axis not in data.columns:
                return {"type": "line", "ok": False, "error": f"Column '{x_axis}' not found"}
            if y_axis not in data.columns:
                return {"type": "line", "ok": False, "error": f"Column '{y_axis}' not found"}
            
            # Sort by x-axis for proper line plotting
            data = data.sort_values(x_axis)
            
            # Sample data for performance
            if len(data) > 1000:
                data = data.sample(1000).sort_values(x_axis)
            
            # Create line plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data[x_axis],
                y=data[y_axis],
                mode='lines+markers',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=4),
                name=f'{x_axis} vs {y_axis}'
            ))
            
            fig.update_layout(
                title=f"{y_axis.title()} vs {x_axis.title()}",
                xaxis_title=x_axis.title(),
                yaxis_title=y_axis.title(),
                template='plotly_dark',
                height=500,
                showlegend=False
            )
            
            return {
                "type": "line",
                "description": f"Line plot of {y_axis} vs {x_axis}",
                "figure": fig.to_json(),
                "data_points": len(data),
                "x_axis": x_axis,
                "y_axis": y_axis,
                "ok": True
            }
            
        except Exception as e:
            logger.error(f"❌ Generic line plot failed: {e}")
            return {"type": "line", "ok": False, "error": str(e)}

    async def _plot_generic_histogram(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a histogram for a specified column"""
        try:
            # Extract column parameter
            column = filters.get('column', filters.get('x', filters.get('parameter', 'temperature')))
            
            # Filter data
            data = getattr(self, "_query_data_override", None)
            if data is None or data.empty:
                data = self.data.copy()
            data = self._apply_filters(data, filters)
            
            if data.empty:
                return {"type": "histogram", "ok": False, "insufficient_data": True, 
                       "reason": f"No data for selected filters.", "filters": self._make_json_safe(filters)}
            
            # Check if column exists
            if column not in data.columns:
                return {"type": "histogram", "ok": False, "error": f"Column '{column}' not found"}
            
            # Create histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=data[column],
                nbinsx=30,
                marker=dict(color='#FF6B6B', opacity=0.7),
                name=column.title()
            ))
            
            fig.update_layout(
                title=f"Distribution of {column.title()}",
                xaxis_title=column.title(),
                yaxis_title="Frequency",
                template='plotly_dark',
                height=500,
                showlegend=False
            )
            
            return {
                "type": "histogram",
                "description": f"Histogram of {column}",
                "figure": fig.to_json(),
                "data_points": len(data),
                "column": column,
                "ok": True
            }
            
        except Exception as e:
            logger.error(f"❌ Generic histogram failed: {e}")
            return {"type": "histogram", "ok": False, "error": str(e)}

    async def _plot_generic_xy(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generic xy plot - alias for scatter plot"""
        return await self._plot_generic_scatter(filters)

    async def _create_generic_plot(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a generic plot based on plot_type parameter"""
        try:
            plot_type = filters.get('plot_type', filters.get('type', 'scatter'))
            
            if plot_type in ['scatter', 'xy']:
                return await self._plot_generic_scatter(filters)
            elif plot_type in ['line', 'profile']:
                return await self._plot_generic_line(filters)
            elif plot_type in ['histogram', 'distribution']:
                return await self._plot_generic_histogram(filters)
            else:
                return {"type": "plot", "ok": False, "error": f"Unknown plot type: {plot_type}"}
                
        except Exception as e:
            logger.error(f"❌ Generic plot creation failed: {e}")
            return {"type": "plot", "ok": False, "error": str(e)}