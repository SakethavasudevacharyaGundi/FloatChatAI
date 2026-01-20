"""
ARGO Ocean Data API - PostgreSQL Enhanced Version
Full PostgreSQL integration with no compromises on functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from contextlib import asynccontextmanager
import asyncpg
import json
import os

# Import AI components  
try:
    from ai.simple_processor import simple_ai
    from ai.advanced_processor import advanced_llm, initialize_advanced_llm
except ImportError:
    # Create dummy classes if AI modules not available
    class DummyAI:
        def is_online(self): return False
        def process_query(self, query, data): return {"analysis": "AI not available"}
    
    simple_ai = DummyAI()
    advanced_llm = DummyAI()
    def initialize_advanced_llm(api_key): pass

try:
    from api.integrated_processor import integrated_processor
except ImportError:
    # If running from api directory, try relative import
    try:
        from integrated_processor import integrated_processor
    except ImportError:
        # Create dummy processor if not available
        class DummyProcessor:
            def is_initialized(self): return False
            async def initialize(self): pass
            async def process_intelligent_query(self, query): 
                return {"error": "Integrated processor not available"}
        integrated_processor = DummyProcessor()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv('DB_HOST', 'localhost'),
    "port": os.getenv('DB_PORT', '5433'), 
    "user": os.getenv('DB_USER', 'postgres'),
    "password": os.getenv('DB_PASSWORD', 'your-database-password-here'),
    "database": os.getenv('DB_NAME', 'argo_ocean_data')
}

# Global database connection pool
db_pool = None

class QueryRequest(BaseModel):
    query: str
    include_visualization: bool = True
    limit: Optional[int] = 50

class QueryResponse(BaseModel):
    query: str
    response: str
    data: Optional[Dict[str, Any]] = None
    processing_time: float
    timestamp: datetime

class DatabaseManager:
    """Manage PostgreSQL database connections and queries"""
    
    @staticmethod
    async def create_pool():
        """Create connection pool"""
        try:
            connection_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            pool = await asyncpg.create_pool(connection_url, min_size=2, max_size=10)
            logger.info("Database connection pool created successfully")
            return pool
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    
    @staticmethod
    async def get_profiles(pool: asyncpg.Pool, limit: int = 50, filters: Dict = None) -> List[Dict]:
        """Get profiles from database"""
        try:
            async with pool.acquire() as conn:
                base_query = """
                    SELECT p.profile_id, p.float_id, p.cycle_number, p.profile_datetime,
                           p.lat, p.lon, p.n_levels, p.summary_json, p.qc_status,
                           f.region, f.status as float_status
                    FROM profiles p
                    LEFT JOIN floats f ON p.float_id = f.float_id
                """
                
                where_conditions = []
                params = []
                param_count = 0
                
                if filters:
                    if 'min_temp' in filters:
                        where_conditions.append(f"(p.summary_json::json->>'avg_temp')::float >= ${param_count + 1}")
                        params.append(filters['min_temp'])
                        param_count += 1
                    
                    if 'max_temp' in filters:
                        where_conditions.append(f"(p.summary_json::json->>'avg_temp')::float <= ${param_count + 1}")
                        params.append(filters['max_temp'])
                        param_count += 1
                    
                    if 'min_depth' in filters:
                        where_conditions.append(f"(p.summary_json::json->>'max_depth')::float >= ${param_count + 1}")
                        params.append(filters['min_depth'])
                        param_count += 1
                    
                    if 'date_start' in filters:
                        where_conditions.append(f"p.profile_datetime >= ${param_count + 1}")
                        params.append(filters['date_start'])
                        param_count += 1
                    
                    if 'date_end' in filters:
                        where_conditions.append(f"p.profile_datetime <= ${param_count + 1}")
                        params.append(filters['date_end'])
                        param_count += 1
                
                if where_conditions:
                    base_query += " WHERE " + " AND ".join(where_conditions)
                
                base_query += f" ORDER BY p.profile_datetime LIMIT ${param_count + 1}"
                params.append(limit)
                
                rows = await conn.fetch(base_query, *params)
                
                profiles = []
                for row in rows:
                    profile = dict(row)
                    # Parse summary JSON
                    if profile['summary_json']:
                        try:
                            profile['summary'] = json.loads(profile['summary_json'])
                        except:
                            profile['summary'] = {}
                    profiles.append(profile)
                
                return profiles
                
        except Exception as e:
            logger.error(f"Error getting profiles: {e}")
            raise
    
    @staticmethod
    async def get_measurements(pool: asyncpg.Pool, profile_ids: List[str] = None, limit: int = 1000) -> List[Dict]:
        """Get measurements from database"""
        try:
            async with pool.acquire() as conn:
                if profile_ids:
                    # Get measurements for specific profiles
                    query = """
                        SELECT m.*, p.profile_datetime, p.lat, p.lon
                        FROM measurements m
                        JOIN profiles p ON m.profile_id = p.profile_id
                        WHERE m.profile_id = ANY($1)
                        ORDER BY m.profile_id, m.depth_m
                        LIMIT $2
                    """
                    rows = await conn.fetch(query, profile_ids, limit)
                else:
                    # Get all measurements
                    query = """
                        SELECT m.*, p.profile_datetime, p.lat, p.lon
                        FROM measurements m
                        JOIN profiles p ON m.profile_id = p.profile_id
                        ORDER BY p.profile_datetime, m.depth_m
                        LIMIT $1
                    """
                    rows = await conn.fetch(query, limit)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting measurements: {e}")
            raise
    
    @staticmethod
    async def search_profiles_by_query(pool: asyncpg.Pool, query: str, limit: int = 50) -> Dict[str, Any]:
        """Search profiles based on query keywords"""
        try:
            query_lower = query.lower()
            filters = {}
            
            # Parse query for filters
            if 'temperature' in query_lower:
                if 'high' in query_lower or 'warm' in query_lower:
                    filters['min_temp'] = 25.0
                elif 'low' in query_lower or 'cold' in query_lower:
                    filters['max_temp'] = 20.0
            
            if 'deep' in query_lower:
                filters['min_depth'] = 1000.0
            elif 'shallow' in query_lower:
                filters['max_depth'] = 200.0
            
            if 'salinity' in query_lower and 'high' in query_lower:
                # We'll need to check measurements for high salinity
                pass
            
            # Get profiles with filters
            profiles = await DatabaseManager.get_profiles(pool, limit, filters)
            
            # Get sample measurements for these profiles
            if profiles:
                profile_ids = [p['profile_id'] for p in profiles[:10]]  # Limit measurements
                measurements = await DatabaseManager.get_measurements(pool, profile_ids, 500)
            else:
                measurements = []
            
            return {
                "profiles": profiles,
                "measurements": measurements,
                "total_profiles": len(profiles),
                "total_measurements": len(measurements),
                "query": query,
                "filters_applied": filters
            }
            
        except Exception as e:
            logger.error(f"Error searching profiles: {e}")
            raise
    
    @staticmethod
    async def get_database_stats(pool: asyncpg.Pool) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            async with pool.acquire() as conn:
                # Get counts
                float_count = await conn.fetchval("SELECT COUNT(*) FROM floats")
                profile_count = await conn.fetchval("SELECT COUNT(*) FROM profiles")
                measurement_count = await conn.fetchval("SELECT COUNT(*) FROM measurements")
                
                # Get date range
                date_range = await conn.fetchrow("""
                    SELECT MIN(profile_datetime) as start_date, MAX(profile_datetime) as end_date
                    FROM profiles
                """)
                
                # Get geographic range
                geo_range = await conn.fetchrow("""
                    SELECT MIN(lat) as lat_min, MAX(lat) as lat_max,
                           MIN(lon) as lon_min, MAX(lon) as lon_max
                    FROM profiles
                """)
                
                # Get temperature range
                temp_range = await conn.fetchrow("""
                    SELECT MIN(temperature_c) as temp_min, MAX(temperature_c) as temp_max,
                           AVG(temperature_c) as temp_avg
                    FROM measurements
                    WHERE temperature_c IS NOT NULL
                """)
                
                # Get depth range
                depth_range = await conn.fetchrow("""
                    SELECT MIN(depth_m) as depth_min, MAX(depth_m) as depth_max,
                           AVG(depth_m) as depth_avg
                    FROM measurements
                """)
                
                return {
                    "counts": {
                        "floats": float_count,
                        "profiles": profile_count,
                        "measurements": measurement_count
                    },
                    "date_range": {
                        "start": date_range['start_date'].isoformat() if date_range['start_date'] else None,
                        "end": date_range['end_date'].isoformat() if date_range['end_date'] else None
                    },
                    "geographic_coverage": {
                        "lat_min": float(geo_range['lat_min']) if geo_range['lat_min'] else None,
                        "lat_max": float(geo_range['lat_max']) if geo_range['lat_max'] else None,
                        "lon_min": float(geo_range['lon_min']) if geo_range['lon_min'] else None,
                        "lon_max": float(geo_range['lon_max']) if geo_range['lon_max'] else None
                    },
                    "temperature_range": {
                        "min": float(temp_range['temp_min']) if temp_range['temp_min'] else None,
                        "max": float(temp_range['temp_max']) if temp_range['temp_max'] else None,
                        "avg": float(temp_range['temp_avg']) if temp_range['temp_avg'] else None
                    },
                    "depth_range": {
                        "min": float(depth_range['depth_min']) if depth_range['depth_min'] else None,
                        "max": float(depth_range['depth_max']) if depth_range['depth_max'] else None,
                        "avg": float(depth_range['depth_avg']) if depth_range['depth_avg'] else None
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global db_pool
    
    # Startup
    logger.info("Starting ARGO Ocean Data API with PostgreSQL...")
    
    try:
        # Create database connection pool
        db_pool = await DatabaseManager.create_pool()
        logger.info("Database connection pool initialized")
        
        # Initialize advanced LLM (check for API key in environment or config)
        try:
            import os
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                try:
                    from config import config
                    api_key = getattr(config, 'GOOGLE_API_KEY', None)
                except ImportError:
                    pass
            
            if api_key:
                initialize_advanced_llm(api_key)
                await advanced_llm.initialize_schema_context()
                logger.info("Advanced LLM initialized with schema context")
            else:
                logger.info("No Google API key found, advanced LLM will use fallback mode")
        except Exception as e:
            logger.warning(f"Advanced LLM initialization failed: {e}")
        
        # Initialize integrated processor for intelligent responses
        try:
            await integrated_processor.initialize()
            logger.info("✅ Integrated processor initialized - RAG + MCP + LLM + Database")
        except Exception as e:
            logger.warning(f"Integrated processor initialization failed: {e}")
        
        # Test database connection
        stats = await DatabaseManager.get_database_stats(db_pool)
        logger.info(f"Database connected successfully: {stats['counts']}")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    
    logger.info("API startup completed successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down ARGO Ocean Data API...")
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")

app = FastAPI(
    title="ARGO Ocean Data API - PostgreSQL Enhanced",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if db_pool:
            # Test database connection
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_status = "connected"
        else:
            db_status = "disconnected"
    except:
        db_status = "error"
    
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "ai_processor": "online" if simple_ai.is_online() else "offline"
    }

@app.get("/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        stats = await DatabaseManager.get_database_stats(db_pool)
        stats["ai_processor_status"] = "online" if simple_ai.is_online() else "offline"
        stats["database_status"] = "connected"
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language queries about ARGO data with integrated intelligence"""
    start_time = datetime.now()
    
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Try integrated processor first (RAG + MCP + LLM + Database)
        if integrated_processor.is_initialized:
            try:
                logger.info("🧠 Using integrated intelligence processor (RAG + MCP + LLM + Database)")
                intelligent_result = await integrated_processor.process_intelligent_query(request.query)
                
                if intelligent_result.get('success'):
                    return QueryResponse(
                        query=request.query,
                        response=intelligent_result['response'],
                        data={
                            "sql_query": intelligent_result['data'].get('sql_query', 'N/A'),
                            "data": intelligent_result['data'].get('data', [])[:request.limit],
                            "total_rows": intelligent_result['data'].get('data_count', 0),
                            "rag_context": intelligent_result.get('rag_context', ''),
                            "mcp_analysis": intelligent_result.get('mcp_analysis', {}),
                            "processing_method": "integrated_intelligence"
                        },
                        processing_time=intelligent_result['processing_time'],
                        timestamp=datetime.now()
                    )
                else:
                    logger.warning(f"Integrated processor failed: {intelligent_result.get('error')}")
            except Exception as e:
                logger.warning(f"Integrated processor failed: {str(e)}")
        
        # Fallback to advanced LLM (if available)
        if advanced_llm.is_online():
            try:
                logger.info("Using advanced LLM with SQL generation")
                llm_result = await advanced_llm.process_advanced_query(request.query)
                
                if llm_result.get('success'):
                    return QueryResponse(
                        query=request.query,
                        response=llm_result['response'],
                        data={
                            "sql_query": llm_result['sql_query'],
                            "data": llm_result['data'][:request.limit],
                            "total_rows": llm_result['data_count'],
                            "processing_method": "advanced_llm_sql"
                        },
                        processing_time=llm_result['processing_time'],
                        timestamp=datetime.now()
                    )
                else:
                    logger.warning(f"Advanced LLM failed: {llm_result.get('error')}")
            except Exception as e:
                logger.warning(f"Advanced LLM processing failed: {str(e)}")
        
        # Final fallback to database search + simple AI
        logger.info("Using fallback database search method")
        query_data = await DatabaseManager.search_profiles_by_query(
            db_pool, request.query, request.limit
        )
        
        # Try to use simple AI processor for better responses
        if simple_ai.is_online():
            try:
                ai_response = await simple_ai.process_query(request.query, query_data)
                response_text = ai_response.get('analysis', f"Analysis of your query: {request.query}")
            except Exception as e:
                logger.warning(f"Simple AI processing failed, using fallback: {str(e)}")
                response_text = generate_fallback_response(request.query, query_data)
        else:
            response_text = generate_fallback_response(request.query, query_data)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            query=request.query,
            response=response_text,
            data=query_data,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_fallback_response(query: str, data: Dict[str, Any]) -> str:
    """Generate fallback response when AI is not available"""
    query_lower = query.lower()
    
    profiles_count = data.get('total_profiles', 0)
    measurements_count = data.get('total_measurements', 0)
    filters = data.get('filters_applied', {})
    
    base_msg = f"Found {profiles_count} profiles with {measurements_count} measurements"
    
    if 'temperature' in query_lower:
        if 'min_temp' in filters:
            return f"{base_msg} with temperatures above {filters['min_temp']}°C. Surface waters show the highest temperatures in the ARGO dataset."
        elif 'max_temp' in filters:
            return f"{base_msg} with temperatures below {filters['max_temp']}°C. Deep waters typically show the lowest temperatures."
        else:
            return f"{base_msg} showing vertical temperature distribution in the Indian Ocean."
    
    elif 'salinity' in query_lower:
        return f"{base_msg} showing salinity measurements and water mass characteristics in the Indian Ocean."
    
    elif 'deep' in query_lower or 'depth' in query_lower:
        if 'min_depth' in filters:
            return f"{base_msg} from depths greater than {filters['min_depth']}m, showing deep ocean conditions."
        else:
            return f"{base_msg} organized by depth levels, showing the vertical structure of the ocean."
    
    elif 'shallow' in query_lower:
        return f"{base_msg} from shallow waters, showing surface and near-surface conditions."
    
    else:
        return f"{base_msg} from ARGO float 1900121 in the Indian Ocean region."

@app.get("/profiles")
async def get_profiles(limit: int = 50):
    """Get ARGO profiles"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        profiles = await DatabaseManager.get_profiles(db_pool, limit)
        return {
            "profiles": profiles,
            "total": len(profiles),
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/measurements")
async def get_measurements(limit: int = 100, profile_id: Optional[str] = None):
    """Get ARGO measurements"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        profile_ids = [profile_id] if profile_id else None
        measurements = await DatabaseManager.get_measurements(db_pool, profile_ids, limit)
        return {
            "measurements": measurements,
            "total": len(measurements),
            "limit": limit,
            "profile_filter": profile_id
        }
    except Exception as e:
        logger.error(f"Error getting measurements: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profiles/{profile_id}")
async def get_profile_details(profile_id: str):
    """Get detailed information for a specific profile"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Get profile info
        profiles = await DatabaseManager.get_profiles(db_pool, 1, {"profile_id": profile_id})
        if not profiles:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        profile = profiles[0]
        
        # Get measurements for this profile
        measurements = await DatabaseManager.get_measurements(db_pool, [profile_id], 1000)
        
        return {
            "profile": profile,
            "measurements": measurements,
            "measurement_count": len(measurements)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting profile details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intelligent-query")
async def intelligent_query(request: QueryRequest):
    """Test endpoint for integrated intelligence processor"""
    if not integrated_processor.is_initialized:
        raise HTTPException(status_code=503, detail="Integrated processor not initialized")
    
    try:
        result = await integrated_processor.process_intelligent_query(request.query)
        return result
    except Exception as e:
        logger.error(f"Intelligent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)