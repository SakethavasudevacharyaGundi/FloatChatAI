import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

class ArgoQueryProcessor:
    def __init__(self):
        self.region_mapping = {
            "indian ocean": "indian_ocean",
            "atlantic ocean": "atlantic_ocean", 
            "pacific ocean": "pacific_ocean",
            "arctic ocean": "arctic_ocean",
            "southern ocean": "southern_ocean",
            "arabian sea": "indian_ocean",
            "bay of bengal": "indian_ocean",
            "equator": "indian_ocean"
        }
        
        self.parameter_mapping = {
            "temperature": "temperature",
            "temp": "temperature",
            "salinity": "salinity",
            "salt": "salinity",
            "oxygen": "oxygen",
            "o2": "oxygen",
            "chlorophyll": "chlorophyll",
            "chl": "chlorophyll",
            "nitrate": "nitrate",
            "no3": "nitrate",
            "ph": "ph",
            "pressure": "pressure",
            "depth": "depth"
        }
        
        self.time_patterns = {
            "last 6 months": 180,
            "last year": 365,
            "last 2 years": 730,
            "last month": 30,
            "last week": 7,
            "recent": 30
        }

    def extract_location_info(self, query: str) -> Dict[str, Any]:
        """Extract location information from query"""
        location_info = {
            "region": None,
            "latitude": None,
            "longitude": None,
            "nearby": False
        }
        
        query_lower = query.lower()
        
        # Check for region mentions
        for region_name, region_code in self.region_mapping.items():
            if region_name in query_lower:
                location_info["region"] = region_code
                break
        
        # Check for coordinate mentions
        lat_pattern = r'(\d+\.?\d*)\s*°?\s*[nNsS]'
        lon_pattern = r'(\d+\.?\d*)\s*°?\s*[eEwW]'
        
        lat_match = re.search(lat_pattern, query)
        lon_match = re.search(lon_pattern, query)
        
        if lat_match and lon_match:
            location_info["latitude"] = float(lat_match.group(1))
            location_info["longitude"] = float(lon_match.group(1))
        
        # Check for "near" or "around" keywords
        if any(word in query_lower for word in ["near", "around", "close to", "nearby"]):
            location_info["nearby"] = True
        
        return location_info

    def extract_parameters(self, query: str) -> List[str]:
        """Extract oceanographic parameters from query"""
        parameters = []
        query_lower = query.lower()
        
        for param_name, param_code in self.parameter_mapping.items():
            if param_name in query_lower:
                parameters.append(param_code)
        
        return parameters

    def extract_time_info(self, query: str) -> Dict[str, Any]:
        """Extract time information from query"""
        time_info = {
            "start_date": None,
            "end_date": None,
            "time_range_days": None
        }
        
        query_lower = query.lower()
        
        # Check for specific time patterns
        for pattern, days in self.time_patterns.items():
            if pattern in query_lower:
                time_info["time_range_days"] = days
                break
        
        # Check for specific month/year mentions
        month_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
        month_match = re.search(month_pattern, query_lower)
        
        if month_match:
            month_name = month_match.group(1)
            year = int(month_match.group(2))
            
            month_map = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12
            }
            
            month_num = month_map[month_name]
            time_info["start_date"] = datetime(year, month_num, 1)
            time_info["end_date"] = datetime(year, month_num + 1, 1) if month_num < 12 else datetime(year + 1, 1, 1)
        
        # Check for year mentions
        year_pattern = r'\b(20\d{2})\b'
        year_matches = re.findall(year_pattern, query)
        if year_matches and not time_info["start_date"]:
            year = int(year_matches[0])
            time_info["start_date"] = datetime(year, 1, 1)
            time_info["end_date"] = datetime(year + 1, 1, 1)
        
        return time_info

    def extract_visualization_type(self, query: str) -> str:
        """Extract visualization type from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["map", "location", "trajectory", "path"]):
            return "map"
        elif any(word in query_lower for word in ["profile", "depth", "vertical"]):
            return "profile"
        elif any(word in query_lower for word in ["compare", "comparison", "vs", "versus"]):
            return "comparison"
        elif any(word in query_lower for word in ["time", "temporal", "trend", "series"]):
            return "time_series"
        else:
            return "table"

    def generate_sql_query(self, location_info: Dict, parameters: List[str], time_info: Dict, limit: int = 100) -> str:
        """Generate SQL query based on extracted information"""
        base_query = """
        SELECT 
            profile_id,
            float_id,
            latitude,
            longitude,
            timestamp,
            depth,
            temperature,
            salinity,
            pressure
        """
        
        # Add BGC parameters if requested
        bgc_params = ["oxygen", "chlorophyll", "nitrate", "ph"]
        for param in bgc_params:
            if param in parameters:
                base_query += f",\n            {param}"
        
        base_query += "\n        FROM argo_profiles\n        WHERE 1=1"
        
        # Add location filters
        if location_info["region"]:
            base_query += f"\n        AND region = '{location_info['region']}'"
        
        if location_info["latitude"] and location_info["longitude"]:
            if location_info["nearby"]:
                # Add proximity filter (within ~100km)
                base_query += f"""
        AND (
            (latitude - {location_info['latitude']})^2 + 
            (longitude - {location_info['longitude']})^2
        ) < 1.0"""
            else:
                base_query += f"\n        AND latitude = {location_info['latitude']}"
                base_query += f"\n        AND longitude = {location_info['longitude']}"
        
        # Add time filters
        if time_info["start_date"] and time_info["end_date"]:
            base_query += f"\n        AND timestamp >= '{time_info['start_date'].isoformat()}'"
            base_query += f"\n        AND timestamp < '{time_info['end_date'].isoformat()}'"
        elif time_info["time_range_days"]:
            base_query += f"\n        AND timestamp >= NOW() - INTERVAL '{time_info['time_range_days']} days'"
        
        # Add parameter filters (only non-null values)
        for param in parameters:
            if param in bgc_params:
                base_query += f"\n        AND {param} IS NOT NULL"
        
        # Add quality filter
        base_query += "\n        AND quality_flag <= 2"  # Good or probably good data
        
        base_query += f"\n        ORDER BY timestamp DESC\n        LIMIT {limit}"
        
        return base_query

    def process_query(self, query: str, limit: int = 100) -> Dict[str, Any]:
        """Main method to process a natural language query"""
        # Extract information from query
        location_info = self.extract_location_info(query)
        parameters = self.extract_parameters(query)
        time_info = self.extract_time_info(query)
        viz_type = self.extract_visualization_type(query)
        
        # Generate SQL query
        sql_query = self.generate_sql_query(location_info, parameters, time_info, limit)
        
        # Create metadata for the query
        metadata = {
            "extracted_location": location_info,
            "extracted_parameters": parameters,
            "extracted_time": time_info,
            "visualization_type": viz_type,
            "query_type": self._classify_query_type(query)
        }
        
        return {
            "original_query": query,
            "sql_query": sql_query,
            "metadata": metadata,
            "visualization_type": viz_type
        }

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["show", "display", "find", "get"]):
            return "retrieval"
        elif any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            return "comparison"
        elif any(word in query_lower for word in ["nearest", "closest", "near", "around"]):
            return "proximity"
        elif any(word in query_lower for word in ["count", "how many", "number"]):
            return "aggregation"
        else:
            return "general"

    def generate_summary(self, query: str, results: List[Dict]) -> str:
        """Generate a natural language summary of the query results"""
        if not results:
            return "No ARGO data found matching your criteria."
        
        num_results = len(results)
        unique_floats = len(set(result.get("float_id", "") for result in results))
        
        # Extract basic stats
        if results:
            first_result = results[0]
            region = first_result.get("region", "unknown")
            date_range = "recent data"
            
            if "timestamp" in first_result:
                try:
                    timestamp = datetime.fromisoformat(first_result["timestamp"].replace("Z", ""))
                    date_range = timestamp.strftime("%B %Y")
                except:
                    pass
        
        summary = f"Found {num_results} ARGO profiles from {unique_floats} different floats"
        
        if region != "unknown":
            summary += f" in the {region.replace('_', ' ').title()}"
        
        summary += f" ({date_range})."
        
        # Add parameter information
        parameters = []
        if any("temperature" in str(result) for result in results):
            parameters.append("temperature")
        if any("salinity" in str(result) for result in results):
            parameters.append("salinity")
        if any("oxygen" in str(result) for result in results):
            parameters.append("oxygen")
        if any("chlorophyll" in str(result) for result in results):
            parameters.append("chlorophyll")
        
        if parameters:
            summary += f" Data includes {', '.join(parameters)} measurements."
        
        return summary
