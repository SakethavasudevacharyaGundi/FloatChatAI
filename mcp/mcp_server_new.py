#!/usr/bin/env python3
"""
ARGO Ocean Data MCP Server

A Model Context Protocol (MCP) server providing comprehensive oceanographic analysis tools
for ARGO float data including data visualization, float tracking, profile analysis, 
and measurement processing capabilities.

This server integrates with PostgreSQL database containing real ARGO oceanographic data
and provides tools for advanced data analysis and visualization generation.
"""

import asyncio
import asyncpg
import json
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Windows
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go

try:
    from rag.rag_pipeline import ArgoRAGPipeline  # semantic search backend
except Exception:
    ArgoRAGPipeline = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArgoMCPServer:
    """
    ARGO Ocean Data MCP Server
    
    Provides comprehensive oceanographic analysis tools including:
    - Float tracking and trajectory analysis
    - Profile data analysis and statistics
    - Measurement processing and quality control
    - Data visualization generation
    - Temporal and spatial analysis
    - Oceanographic parameter correlations
    """
    
    def __init__(self):
        """Initialize the ARGO MCP Server"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5433')),
            'database': os.getenv('DB_NAME', 'argo_ocean_data'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'your-database-password-here')
        }
        self.db_pool = None
        self.tools = {}
        self._register_tools()
        
        logger.info("ARGO MCP Server initialized")
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                min_size=5,
                max_size=20
            )
            logger.info("Database connection pool created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {str(e)}")
            return False
    
    def _register_tools(self):
        """Register all available MCP tools"""
        
        # Float Analysis Tools
        self.tools['get_float_trajectory'] = {
            'description': 'Get complete trajectory and movement pattern of an ARGO float',
            'parameters': {
                'float_id': {'type': 'string', 'description': 'ARGO float identifier'},
                'start_date': {'type': 'string', 'description': 'Start date (YYYY-MM-DD) - optional'},
                'end_date': {'type': 'string', 'description': 'End date (YYYY-MM-DD) - optional'}
            }
        }
        
        self.tools['analyze_float_performance'] = {
            'description': 'Analyze float operational performance including profile frequency, data quality, and sensor status',
            'parameters': {
                'float_id': {'type': 'string', 'description': 'ARGO float identifier'}
            }
        }
        
        # Profile Analysis Tools
        self.tools['get_profile_details'] = {
            'description': 'Get detailed information about a specific oceanographic profile',
            'parameters': {
                'profile_id': {'type': 'string', 'description': 'Profile identifier'}
            }
        }
        
        self.tools['compare_profiles'] = {
            'description': 'Compare multiple oceanographic profiles for similarities and differences',
            'parameters': {
                'profile_ids': {'type': 'array', 'description': 'List of profile identifiers to compare'},
                'parameters': {'type': 'array', 'description': 'Parameters to compare (temperature, salinity, etc.)'}
            }
        }
        
        self.tools['find_similar_profiles'] = {
            'description': 'Find profiles with similar characteristics (location, time, or oceanographic properties)',
            'parameters': {
                'reference_profile_id': {'type': 'string', 'description': 'Reference profile identifier'},
                'similarity_criteria': {'type': 'string', 'description': 'Criteria: geographic, temporal, or oceanographic'},
                'threshold': {'type': 'number', 'description': 'Similarity threshold (0-1)'}
            }
        }
        
        # Measurement Processing Tools
        self.tools['analyze_depth_profile'] = {
            'description': 'Analyze oceanographic measurements at different depths for a specific profile',
            'parameters': {
                'profile_id': {'type': 'string', 'description': 'Profile identifier'},
                'parameters': {'type': 'array', 'description': 'Parameters to analyze (temperature_c, salinity_psu, etc.)'}
            }
        }
        
        self.tools['detect_anomalies'] = {
            'description': 'Detect anomalous measurements in oceanographic data',
            'parameters': {
                'float_id': {'type': 'string', 'description': 'Float identifier - optional'},
                'parameter': {'type': 'string', 'description': 'Parameter to check (temperature_c, salinity_psu, etc.)'},
                'time_window': {'type': 'integer', 'description': 'Time window in days for comparison'}
            }
        }
        
        self.tools['calculate_water_mass_properties'] = {
            'description': 'Calculate water mass properties including density, potential temperature, and mixed layer depth',
            'parameters': {
                'profile_id': {'type': 'string', 'description': 'Profile identifier'}
            }
        }
        
        # Spatial Analysis Tools
        self.tools['get_regional_statistics'] = {
            'description': 'Get oceanographic statistics for a specific geographic region',
            'parameters': {
                'min_lat': {'type': 'number', 'description': 'Minimum latitude'},
                'max_lat': {'type': 'number', 'description': 'Maximum latitude'},
                'min_lon': {'type': 'number', 'description': 'Minimum longitude'},
                'max_lon': {'type': 'number', 'description': 'Maximum longitude'},
                'parameter': {'type': 'string', 'description': 'Parameter to analyze'},
                'depth_range': {'type': 'array', 'description': 'Depth range [min, max] in meters'}
            }
        }
        
        self.tools['track_water_mass_movement'] = {
            'description': 'Track movement and evolution of water masses across multiple profiles',
            'parameters': {
                'region': {'type': 'string', 'description': 'Ocean region or geographic bounds'},
                'time_period': {'type': 'array', 'description': 'Time period [start_date, end_date]'},
                'water_mass_criteria': {'type': 'object', 'description': 'Water mass identification criteria'}
            }
        }
        
        # Temporal Analysis Tools  
        self.tools['analyze_seasonal_patterns'] = {
            'description': 'Analyze seasonal patterns in oceanographic data',
            'parameters': {
                'region': {'type': 'string', 'description': 'Geographic region or specific coordinates'},
                'parameter': {'type': 'string', 'description': 'Parameter to analyze'},
                'years': {'type': 'array', 'description': 'Years to include in analysis'}
            }
        }
        
        self.tools['detect_temporal_trends'] = {
            'description': 'Detect long-term trends in oceanographic parameters',
            'parameters': {
                'parameter': {'type': 'string', 'description': 'Parameter to analyze'},
                'region': {'type': 'string', 'description': 'Geographic region'},
                'depth_range': {'type': 'array', 'description': 'Depth range [min, max] in meters'}
            }
        }
        
        # Visualization Tools
        self.tools['generate_temperature_profile_plot'] = {
            'description': 'Generate temperature vs depth profile visualization',
            'parameters': {
                'profile_ids': {'type': 'array', 'description': 'List of profile identifiers'},
                'comparison_mode': {'type': 'boolean', 'description': 'Whether to overlay multiple profiles'}
            }
        }
        
        self.tools['generate_geographic_map'] = {
            'description': 'Generate geographic map showing float positions and data',
            'parameters': {
                'float_ids': {'type': 'array', 'description': 'List of float identifiers'},
                'parameter': {'type': 'string', 'description': 'Parameter to visualize on map'},
                'time_range': {'type': 'array', 'description': 'Time range [start, end]'}
            }
        }
        
        self.tools['generate_time_series_plot'] = {
            'description': 'Generate time series visualization of oceanographic parameters',
            'parameters': {
                'float_id': {'type': 'string', 'description': 'Float identifier'},
                'parameter': {'type': 'string', 'description': 'Parameter to plot'},
                'depth_filter': {'type': 'number', 'description': 'Specific depth level (optional)'}
            }
        }
        
        self.tools['generate_correlation_matrix'] = {
            'description': 'Generate correlation matrix heatmap for oceanographic parameters',
            'parameters': {
                'profile_ids': {'type': 'array', 'description': 'List of profile identifiers'},
                'parameters': {'type': 'array', 'description': 'Parameters to include in correlation analysis'}
            }
        }
        
        # Data Quality Tools
        self.tools['assess_data_quality'] = {
            'description': 'Assess quality of ARGO data including completeness, accuracy, and consistency',
            'parameters': {
                'float_id': {'type': 'string', 'description': 'Float identifier - optional'},
                'profile_id': {'type': 'string', 'description': 'Profile identifier - optional'},
                'quality_metrics': {'type': 'array', 'description': 'Specific quality metrics to assess'}
            }
        }
        
        self.tools['recommend_data_filters'] = {
            'description': 'Recommend appropriate data filters based on quality control flags and statistical analysis',
            'parameters': {
                'dataset_description': {'type': 'string', 'description': 'Description of intended data usage'},
                'quality_requirements': {'type': 'string', 'description': 'Required quality level (high, medium, low)'}
            }
        }

        # Generic minimal tool surface for LLM-directed orchestration
        self.tools['sql_query'] = {
            'description': 'Run a validated SQL query against PostgreSQL with whitelist and timeout',
            'parameters': {
                'table': {'type': 'string'},
                'columns': {'type': 'array'},
                'filters': {'type': 'object'},
                'aggregations': {'type': 'array'},
                'limit': {'type': 'integer'}
            }
        }
        self.tools['visualize'] = {
            'description': 'Build a Plotly visualization from provided data snapshot/params',
            'parameters': {
                'type': {'type': 'string'},
                'params': {'type': 'object'}
            }
        }
        self.tools['semantic_search'] = {
            'description': 'Semantic search over enhanced RAG corpus and return top-k chunks',
            'parameters': {
                'query': {'type': 'string'},
                'k': {'type': 'integer'}
            }
        }
        
        logger.info(f"Registered {len(self.tools)} MCP tools")
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific MCP tool with given parameters"""
        
        if tool_name not in self.tools:
            return {
                'success': False,
                'error': f"Unknown tool: {tool_name}",
                'available_tools': list(self.tools.keys())
            }
        
        try:
            # Route to appropriate handler method
            handler_name = f"_handle_{tool_name}"
            if hasattr(self, handler_name):
                handler = getattr(self, handler_name)
                # Enforce per-tool timeout
                result = await asyncio.wait_for(handler(parameters), timeout=20)
                
                return {
                    'success': True,
                    'tool': tool_name,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f"Handler not implemented for tool: {tool_name}",
                    'tool': tool_name
                }
                
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name
            }
    
    # Tool Handler Methods

    # ---- Minimal generic tools ----
    def _allowed_schema(self) -> Dict[str, List[str]]:
        return {
            'floats': ['float_id','model','platform_number','deploy_date','last_lat','last_lon','region','status','total_profiles','last_profile_date','created_at','updated_at'],
            'profiles': ['profile_id','float_id','cycle_number','profile_datetime','lat','lon','qc_status','created_at','updated_at'],
            'measurements': ['meas_id','profile_id','depth_m','temperature_c','salinity_psu','pressure_dbar','created_at'],
            'profile_summaries': ['profile_id','measurement_count','min_depth','max_depth','avg_temperature','avg_salinity','created_at','updated_at']
        }

    async def _handle_sql_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        table = str(params.get('table','')).strip()
        columns = params.get('columns') or []
        filters = params.get('filters') or {}
        aggregations = params.get('aggregations') or []
        limit = int(params.get('limit') or 100)

        schema = self._allowed_schema()
        if table not in schema:
            raise ValueError(f"Table not allowed: {table}")

        # Validate columns
        allowed_cols = set(schema[table])
        proj_cols: List[str] = []
        for c in columns:
            c = str(c).strip()
            if c not in allowed_cols:
                raise ValueError(f"Column not allowed: {c}")
            proj_cols.append(c)
        if not proj_cols and not aggregations:
            proj_cols = list(allowed_cols)[:5]

        # Validate aggregations
        agg_sql_parts: List[str] = []
        agg_allowed = {'avg','min','max','count'}
        for agg in aggregations:
            fn = str(agg.get('function','')).lower()
            col = str(agg.get('column','')).strip()
            alias = str(agg.get('alias', f"{fn}_{col}"))
            if fn not in agg_allowed:
                raise ValueError(f"Aggregation not allowed: {fn}")
            if col not in allowed_cols:
                raise ValueError(f"Aggregation column not allowed: {col}")
            agg_sql_parts.append(f"{fn}({col}) AS {alias}")

        select_list = ", ".join(proj_cols) if proj_cols else ""
        if agg_sql_parts:
            select_list = (select_list + (", " if select_list else "")) + ", ".join(agg_sql_parts)
        if not select_list:
            select_list = "*"

        # WHERE with parameterization
        where_sql = ["TRUE"]
        values: List[Any] = []
        idx = 1
        for key, val in filters.items():
            if key not in allowed_cols:
                raise ValueError(f"Filter column not allowed: {key}")
            if isinstance(val, dict):
                op = str(val.get('op','eq')).lower()
                if op == 'eq':
                    where_sql.append(f"{key} = ${idx}")
                    values.append(val.get('value'))
                    idx += 1
                elif op == 'gte':
                    where_sql.append(f"{key} >= ${idx}")
                    values.append(val.get('value'))
                    idx += 1
                elif op == 'lte':
                    where_sql.append(f"{key} <= ${idx}")
                    values.append(val.get('value'))
                    idx += 1
                elif op == 'between':
                    lo, hi = val.get('low'), val.get('high')
                    where_sql.append(f"{key} BETWEEN ${idx} AND ${idx+1}")
                    values.extend([lo, hi])
                    idx += 2
                else:
                    raise ValueError(f"Operator not allowed: {op}")
            else:
                where_sql.append(f"{key} = ${idx}")
                values.append(val)
                idx += 1

        sql = f"SELECT {select_list} FROM {table} WHERE {' AND '.join(where_sql)} LIMIT {min(limit, 1000)}"
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(sql, *values)
            recs = [dict(r) for r in rows]
            # cast numpy/pandas types
            for r in recs:
                for k,v in list(r.items()):
                    if isinstance(v, (np.floating, np.integer)):
                        r[k] = float(v) if isinstance(v, np.floating) else int(v)
                    elif hasattr(v, 'isoformat'):
                        try:
                            r[k] = v.isoformat()
                        except Exception:
                            pass
            return { 'sql': sql, 'row_count': len(recs), 'rows': recs }

    async def _handle_visualize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        vtype = str(params.get('type','')).lower()
        p = params.get('params') or {}
        fig = go.Figure()
        if vtype == 'scatter' and 'x' in p and 'y' in p:
            fig.add_trace(go.Scatter(x=p.get('x'), y=p.get('y'), mode='markers', marker=dict(size=6)))
        elif vtype == 'line' and 'x' in p and 'y' in p:
            fig.add_trace(go.Scatter(x=p.get('x'), y=p.get('y'), mode='lines'))
        elif vtype == 'map' and {'lat','lon'} <= set(p.keys()):
            fig.add_trace(go.Scattergeo(lat=p.get('lat'), lon=p.get('lon'), mode='markers', marker=dict(size=5)))
            fig.update_geos(projection_type='natural earth')
        else:
            return {'error':'Unsupported visualization type or missing params'}
        return {'type': vtype, 'figure': json.dumps(fig, cls=PlotlyJSONEncoder)}

    async def _handle_semantic_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = str(params.get('query','')).strip()
        k = int(params.get('k') or 5)
        if not query:
            return {'error':'query is required'}
        if ArgoRAGPipeline is None:
            return {'error':'RAG pipeline unavailable'}
        rag = ArgoRAGPipeline()
        results = await rag.search_enhanced_rag(query, top_k=k, similarity_threshold=0.3)
        return { 'results': results }
    
    async def _handle_get_float_trajectory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get complete trajectory of an ARGO float"""
        
        float_id = params.get('float_id')
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT 
                    p.profile_id,
                    p.profile_datetime,
                    p.lat,
                    p.lon,
                    p.cycle_number,
                    ps.measurement_count,
                    ps.max_depth,
                    ps.avg_temperature,
                    ps.avg_salinity
                FROM profiles p
                LEFT JOIN profile_summaries ps ON p.profile_id = ps.profile_id
                WHERE p.float_id = $1
            """
            
            params_list = [float_id]
            
            if start_date:
                query += " AND p.profile_datetime >= $2"
                params_list.append(start_date)
                
            if end_date:
                if start_date:
                    query += " AND p.profile_datetime <= $3"
                else:
                    query += " AND p.profile_datetime <= $2"
                params_list.append(end_date)
            
            query += " ORDER BY p.profile_datetime"
            
            rows = await conn.fetch(query, *params_list)
            
            if not rows:
                return {
                    'float_id': float_id,
                    'trajectory': [],
                    'summary': 'No trajectory data found'
                }
            
            trajectory = []
            for row in rows:
                trajectory.append({
                    'profile_id': row['profile_id'],
                    'datetime': row['profile_datetime'].isoformat(),
                    'latitude': float(row['lat']),
                    'longitude': float(row['lon']),
                    'cycle_number': row['cycle_number'],
                    'measurement_count': row['measurement_count'],
                    'max_depth': float(row['max_depth']) if row['max_depth'] else None,
                    'avg_temperature': float(row['avg_temperature']) if row['avg_temperature'] else None,
                    'avg_salinity': float(row['avg_salinity']) if row['avg_salinity'] else None
                })
            
            # Calculate trajectory statistics
            lats = [p['latitude'] for p in trajectory]
            lons = [p['longitude'] for p in trajectory]
            
            total_distance = 0
            for i in range(1, len(trajectory)):
                # Simple distance calculation (could be improved with haversine)
                lat_diff = trajectory[i]['latitude'] - trajectory[i-1]['latitude']
                lon_diff = trajectory[i]['longitude'] - trajectory[i-1]['longitude']
                distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Approximate km
                total_distance += distance
            
            summary = {
                'float_id': float_id,
                'total_profiles': len(trajectory),
                'time_span': {
                    'start': trajectory[0]['datetime'],
                    'end': trajectory[-1]['datetime']
                },
                'geographic_range': {
                    'lat_min': min(lats),
                    'lat_max': max(lats),
                    'lon_min': min(lons),
                    'lon_max': max(lons)
                },
                'estimated_distance_km': round(total_distance, 2),
                'drift_rate_km_per_day': round(total_distance / max(1, (datetime.fromisoformat(trajectory[-1]['datetime']) - datetime.fromisoformat(trajectory[0]['datetime'])).days), 3)
            }
            
            return {
                'float_id': float_id,
                'trajectory': trajectory,
                'summary': summary
            }
    
    async def _handle_analyze_float_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze float operational performance"""
        
        float_id = params.get('float_id')
        
        async with self.db_pool.acquire() as conn:
            # Get float basic info
            float_info = await conn.fetchrow("""
                SELECT * FROM floats WHERE float_id = $1
            """, float_id)
            
            if not float_info:
                return {
                    'error': f'Float {float_id} not found'
                }
            
            # Get profile statistics
            profile_stats = await conn.fetch("""
                SELECT 
                    COUNT(*) as total_profiles,
                    MIN(profile_datetime) as first_profile,
                    MAX(profile_datetime) as last_profile,
                    AVG(n_levels) as avg_levels_per_profile,
                    COUNT(CASE WHEN qc_status = 'good' THEN 1 END) as good_quality_profiles,
                    COUNT(CASE WHEN qc_status = 'questionable' THEN 1 END) as questionable_profiles,
                    COUNT(CASE WHEN qc_status = 'bad' THEN 1 END) as bad_profiles
                FROM profiles 
                WHERE float_id = $1
            """, float_id)
            
            # Get measurement statistics
            measurement_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_measurements,
                    COUNT(CASE WHEN m.temperature_c IS NOT NULL THEN 1 END) as temp_measurements,
                    COUNT(CASE WHEN m.salinity_psu IS NOT NULL THEN 1 END) as salinity_measurements,
                    COUNT(CASE WHEN m.pressure_dbar IS NOT NULL THEN 1 END) as pressure_measurements,
                    COUNT(CASE WHEN m.qc_flags = 'good' THEN 1 END) as good_measurements,
                    AVG(m.depth_m) as avg_measurement_depth,
                    MAX(m.depth_m) as max_measurement_depth
                FROM measurements m
                JOIN profiles p ON m.profile_id = p.profile_id
                WHERE p.float_id = $1
            """, float_id)
            
            # Calculate performance metrics
            stats = profile_stats[0]
            operational_days = (stats['last_profile'] - stats['first_profile']).days
            profile_frequency = stats['total_profiles'] / max(1, operational_days) if operational_days > 0 else 0
            
            data_completeness = {
                'temperature': (measurement_stats['temp_measurements'] / measurement_stats['total_measurements'] * 100) if measurement_stats['total_measurements'] > 0 else 0,
                'salinity': (measurement_stats['salinity_measurements'] / measurement_stats['total_measurements'] * 100) if measurement_stats['total_measurements'] > 0 else 0,
                'pressure': (measurement_stats['pressure_measurements'] / measurement_stats['total_measurements'] * 100) if measurement_stats['total_measurements'] > 0 else 0
            }
            
            quality_assessment = {
                'profile_quality': {
                    'good': stats['good_quality_profiles'],
                    'questionable': stats['questionable_profiles'],
                    'bad': stats['bad_profiles'],
                    'good_percentage': (stats['good_quality_profiles'] / stats['total_profiles'] * 100) if stats['total_profiles'] > 0 else 0
                },
                'measurement_quality': {
                    'good_measurements': measurement_stats['good_measurements'],
                    'total_measurements': measurement_stats['total_measurements'],
                    'good_percentage': (measurement_stats['good_measurements'] / measurement_stats['total_measurements'] * 100) if measurement_stats['total_measurements'] > 0 else 0
                }
            }
            
            performance_analysis = {
                'float_info': dict(float_info),
                'operational_summary': {
                    'total_profiles': stats['total_profiles'],
                    'operational_period_days': operational_days,
                    'first_profile': stats['first_profile'].isoformat(),
                    'last_profile': stats['last_profile'].isoformat(),
                    'profile_frequency_per_day': round(profile_frequency, 3),
                    'avg_levels_per_profile': round(float(stats['avg_levels_per_profile']), 1)
                },
                'data_completeness': data_completeness,
                'quality_assessment': quality_assessment,
                'measurement_summary': {
                    'total_measurements': measurement_stats['total_measurements'],
                    'avg_depth': round(float(measurement_stats['avg_measurement_depth']), 1) if measurement_stats['avg_measurement_depth'] else None,
                    'max_depth': round(float(measurement_stats['max_measurement_depth']), 1) if measurement_stats['max_measurement_depth'] else None
                }
            }
            
            # Performance rating
            overall_score = (
                (quality_assessment['profile_quality']['good_percentage'] * 0.3) +
                (quality_assessment['measurement_quality']['good_percentage'] * 0.3) +
                (min(data_completeness['temperature'], 100) * 0.2) +
                (min(data_completeness['salinity'], 100) * 0.2)
            )
            
            if overall_score >= 80:
                performance_rating = "Excellent"
            elif overall_score >= 60:
                performance_rating = "Good"
            elif overall_score >= 40:
                performance_rating = "Fair"
            else:
                performance_rating = "Poor"
            
            performance_analysis['overall_rating'] = {
                'score': round(overall_score, 1),
                'rating': performance_rating
            }
            
            return performance_analysis
    
    async def _handle_generate_temperature_profile_plot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate temperature vs depth profile visualization"""
        
        profile_ids = params.get('profile_ids', [])
        comparison_mode = params.get('comparison_mode', False)
        
        if not profile_ids:
            return {'error': 'No profile IDs provided'}
        
        async with self.db_pool.acquire() as conn:
            plot_data = []
            
            for profile_id in profile_ids:
                # Get measurements for profile
                measurements = await conn.fetch("""
                    SELECT 
                        m.depth_m,
                        m.temperature_c,
                        p.profile_datetime,
                        p.lat,
                        p.lon
                    FROM measurements m
                    JOIN profiles p ON m.profile_id = p.profile_id
                    WHERE m.profile_id = $1 
                    AND m.temperature_c IS NOT NULL
                    ORDER BY m.depth_m
                """, profile_id)
                
                if measurements:
                    plot_data.append({
                        'profile_id': profile_id,
                        'depths': [float(m['depth_m']) for m in measurements],
                        'temperatures': [float(m['temperature_c']) for m in measurements],
                        'datetime': measurements[0]['profile_datetime'].isoformat(),
                        'location': f"{measurements[0]['lat']:.2f}, {measurements[0]['lon']:.2f}"
                    })
            
            if not plot_data:
                return {'error': 'No temperature data found for provided profiles'}
            
            # Create matplotlib figure
            plt.figure(figsize=(10, 8))
            
            if comparison_mode and len(plot_data) > 1:
                colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
                for i, profile_data in enumerate(plot_data):
                    plt.plot(profile_data['temperatures'], profile_data['depths'], 
                            color=colors[i], linewidth=2, 
                            label=f"{profile_data['profile_id']} ({profile_data['datetime'][:10]})")
                plt.legend()
            else:
                profile_data = plot_data[0]
                plt.plot(profile_data['temperatures'], profile_data['depths'], 
                        'b-', linewidth=2, marker='o', markersize=4)
                plt.title(f"Temperature Profile: {profile_data['profile_id']}\n"
                         f"Date: {profile_data['datetime'][:10]}, Location: {profile_data['location']}")
            
            plt.xlabel('Temperature (°C)')
            plt.ylabel('Depth (m)')
            plt.gca().invert_yaxis()  # Invert y-axis so depth increases downward
            plt.grid(True, alpha=0.3)
            
            if comparison_mode and len(plot_data) > 1:
                plt.title(f'Temperature Profile Comparison ({len(plot_data)} profiles)')
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Generate analysis summary
            all_temps = []
            depth_ranges = []
            for profile_data in plot_data:
                all_temps.extend(profile_data['temperatures'])
                depth_ranges.append({
                    'profile': profile_data['profile_id'],
                    'min_depth': min(profile_data['depths']),
                    'max_depth': max(profile_data['depths']),
                    'temp_range': [min(profile_data['temperatures']), max(profile_data['temperatures'])]
                })
            
            analysis = {
                'temperature_range': [min(all_temps), max(all_temps)],
                'profiles_analyzed': len(plot_data),
                'depth_ranges': depth_ranges
            }
            
            return {
                'plot_base64': plot_base64,
                'plot_type': 'temperature_profile',
                'analysis': analysis,
                'profiles_included': profile_ids
            }
    
    async def get_tool_list(self) -> List[Dict[str, Any]]:
        """Get list of all available tools with descriptions"""
        return [
            {
                'name': name,
                'description': info['description'],
                'parameters': info['parameters']
            }
            for name, info in self.tools.items()
        ]
    
    async def close(self):
        """Close database connections"""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("Database connection pool closed")

# Example usage and testing
async def test_mcp_server():
    """Test the MCP server functionality"""
    
    server = ArgoMCPServer()
    
    if not await server.initialize():
        logger.error("Failed to initialize MCP server")
        return
    
    logger.info("=== Testing ARGO MCP Server ===")
    
    # Test tool listing
    tools = await server.get_tool_list()
    logger.info(f"Available tools: {len(tools)}")
    
    # Test float trajectory
    logger.info("Testing float trajectory tool...")
    trajectory_result = await server.execute_tool('get_float_trajectory', {
        'float_id': '1900121'
    })
    
    if trajectory_result['success']:
        summary = trajectory_result['result']['summary']
        logger.info(f"✅ Float trajectory: {summary['total_profiles']} profiles, "
                   f"{summary['estimated_distance_km']} km traveled")
    
    # Test float performance analysis
    logger.info("Testing float performance analysis...")
    performance_result = await server.execute_tool('analyze_float_performance', {
        'float_id': '1900121'
    })
    
    if performance_result['success']:
        rating = performance_result['result']['overall_rating']
        logger.info(f"✅ Float performance: {rating['rating']} ({rating['score']}/100)")
    
    # Test visualization generation
    logger.info("Testing temperature profile visualization...")
    viz_result = await server.execute_tool('generate_temperature_profile_plot', {
        'profile_ids': ['1900121_000', '1900121_001'],
        'comparison_mode': True
    })
    
    if viz_result['success']:
        analysis = viz_result['result']['analysis']
        logger.info(f"✅ Visualization: {analysis['profiles_analyzed']} profiles, "
                   f"temp range {analysis['temperature_range'][0]:.1f}-{analysis['temperature_range'][1]:.1f}°C")
    
    await server.close()
    logger.info("=== MCP Server Testing Complete ===")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())