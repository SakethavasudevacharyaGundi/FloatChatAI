"""
Enhanced MCP Client for ARGO Ocean Data System
Provides seamless integration between Gemini AI and MCP tools
"""

import json
import asyncio
import subprocess
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MCPToolCall:
    """Represents an MCP tool call"""
    name: str
    arguments: Dict[str, Any]
    
@dataclass
class MCPResponse:
    """Represents an MCP response"""
    success: bool
    data: Any = None
    error: str = None

class ArgoMCPClient:
    """Enhanced MCP Client for ARGO system integration"""
    
    def __init__(self, production_mode: bool = True):
        """
        Initialize MCP Client
        
        Args:
            production_mode: If True, connects to real MCP server. If False, uses mock mode
        """
        self.server_process = None
        self.available_tools = {}
        self.initialized = False
        self.production_mode = production_mode
        self.server_instance = None
        
    async def initialize(self) -> bool:
        """Initialize MCP connection"""
        try:
            if self.production_mode:
                # Production mode: Connect to real MCP server
                logger.info("Initializing MCP client in PRODUCTION mode")
                return await self._initialize_production_mode()
            else:
                # Mock mode: Use simulated tools
                logger.info("Initializing MCP client in MOCK mode")
                return await self._initialize_mock_mode()
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            return False
    
    async def _initialize_production_mode(self) -> bool:
        """Initialize connection to real MCP server"""
        try:
            # Import and initialize the real MCP server
            from mcp.mcp_server import ArgoMCPServer
            
            logger.info("Creating MCP server instance...")
            self.server_instance = ArgoMCPServer()
            
            # Initialize the server (connects to database)
            logger.info("Initializing MCP server connection to database...")
            db_initialized = await self.server_instance.initialize()
            
            if not db_initialized:
                logger.error("Failed to initialize MCP server database connection")
                return False
            
            # Get available tools from the server
            logger.info("Loading available MCP tools...")
            tools_list = await self.server_instance.get_tool_list()
            
            for tool in tools_list:
                self.available_tools[tool['name']] = {
                    'name': tool['name'],
                    'description': tool['description'],
                    'parameters': tool['parameters']
                }
            
            self.initialized = True
            logger.info(f"✅ MCP Client initialized in PRODUCTION mode with {len(self.available_tools)} tools")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import MCP server: {e}")
            logger.warning("Falling back to mock mode")
            self.production_mode = False
            return await self._initialize_mock_mode()
        except Exception as e:
            logger.error(f"Failed to initialize production mode: {e}")
            logger.warning("Falling back to mock mode")
            self.production_mode = False
            return await self._initialize_mock_mode()
    
    async def _initialize_mock_mode(self) -> bool:
        """Initialize mock mode with simulated tools"""
        try:
            logger.info("Initializing MCP client in mock mode")
            
            # Mock available tools
            self.available_tools = {
                "query_argo_data": {
                    "name": "query_argo_data",
                    "description": "Query ARGO data using natural language",
                    "parameters": {"query": "string", "limit": "number"}
                },
                "search_profiles": {
                    "name": "search_profiles", 
                    "description": "Search for ARGO profiles",
                    "parameters": {"query": "string", "n_results": "number"}
                },
                "search_floats": {
                    "name": "search_floats",
                    "description": "Search for ARGO floats", 
                    "parameters": {"query": "string", "n_results": "number"}
                },
                "get_available_regions": {
                    "name": "get_available_regions",
                    "description": "Get available ocean regions",
                    "parameters": {}
                },
                "get_available_parameters": {
                    "name": "get_available_parameters", 
                    "description": "Get available oceanographic parameters",
                    "parameters": {}
                },
                "get_system_stats": {
                    "name": "get_system_stats",
                    "description": "Get system statistics",
                    "parameters": {}
                }
            }
            
            self.initialized = True
            logger.info(f"MCP Client initialized with {len(self.available_tools)} tools (mock mode)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            return False
    
    async def list_tools(self) -> MCPResponse:
        """List all available MCP tools"""
        if not self.initialized:
            await self.initialize()
        
        tools_list = [
            {
                "name": name,
                "description": tool_info.get("description", ""),
                "parameters": tool_info.get("parameters", {})
            }
            for name, tool_info in self.available_tools.items()
        ]
        
        return MCPResponse(success=True, data={"tools": tools_list})
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> MCPResponse:
        """Call a specific MCP tool"""
        if not self.initialized:
            await self.initialize()
        
        if tool_name not in self.available_tools:
            return MCPResponse(
                success=False,
                error=f"Tool '{tool_name}' not available. Available tools: {list(self.available_tools.keys())}"
            )
        
        try:
            if self.production_mode and self.server_instance:
                # Production mode: Call real MCP server
                logger.info(f"Calling MCP tool in PRODUCTION mode: {tool_name}")
                result = await self.server_instance.execute_tool(tool_name, arguments or {})
                
                if result.get('success'):
                    return MCPResponse(success=True, data=result.get('result'))
                else:
                    return MCPResponse(success=False, error=result.get('error', 'Unknown error'))
            else:
                # Mock mode: Use simulated responses
                logger.info(f"Calling MCP tool in MOCK mode: {tool_name}")
                mock_data = await self._get_mock_response(tool_name, arguments or {})
                return MCPResponse(success=True, data=mock_data)
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return MCPResponse(success=False, error=str(e))
    
    async def query_argo_data(self, query: str, limit: int = 100) -> MCPResponse:
        """Query ARGO data using natural language"""
        return await self.call_tool("query_argo_data", {
            "query": query,
            "limit": limit
        })
    
    async def search_profiles(self, query: str, n_results: int = 10) -> MCPResponse:
        """Search for ARGO profiles"""
        return await self.call_tool("search_profiles", {
            "query": query,
            "n_results": n_results
        })
    
    async def search_floats(self, query: str, n_results: int = 10) -> MCPResponse:
        """Search for ARGO floats"""
        return await self.call_tool("search_floats", {
            "query": query,
            "n_results": n_results
        })
    
    async def get_available_regions(self) -> MCPResponse:
        """Get available ocean regions"""
        return await self.call_tool("get_available_regions")
    
    async def get_available_parameters(self) -> MCPResponse:
        """Get available oceanographic parameters"""
        return await self.call_tool("get_available_parameters")
    
    async def get_system_stats(self) -> MCPResponse:
        """Get system statistics"""
        return await self.call_tool("get_system_stats")
    
    async def _get_mock_response(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Generate responses using REAL ARGO data from CSV"""
        from data.real_argo_data_loader import real_argo_loader
        from datetime import datetime, timedelta
        import random
        
        try:
            if tool_name == "query_argo_data":
                query = arguments.get("query", "")
                limit = arguments.get("limit", 20)
                
                logger.info(f"Querying real ARGO data: '{query}' (limit: {limit})")
                
                # Use real data search
                profiles = real_argo_loader.search_profiles(query, limit)
                
                # Convert to response format
                profile_data = []
                for profile in profiles[:limit]:
                    profile_dict = profile.to_dict()
                    profile_data.append(profile_dict)
                
                return {
                    "query": query,
                    "profiles": profile_data,
                    "total_found": len(profile_data),
                    "data_source": "real_argo_csv",
                    "float_id": "1900121"
                }
            
            elif tool_name == "search_profiles":
                query = arguments.get("query", "")
                n_results = arguments.get("n_results", 10)
                
                logger.info(f"Searching real ARGO profiles: '{query}' (limit: {n_results})")
                
                profiles = real_argo_loader.search_profiles(query, n_results)
                
                return {
                    "query": query,
                    "profiles": [profile.to_dict() for profile in profiles],
                    "total_found": len(profiles),
                    "data_source": "real_argo_csv"
                }
            
            elif tool_name == "search_floats":
                query = arguments.get("query", "")
                n_results = arguments.get("n_results", 5)
                
                # Get unique float information from real data
                stats = real_argo_loader.get_profile_statistics()
                
                return {
                    "query": query,
                    "floats": [{
                        "float_id": "1900121",
                        "status": "completed",
                        "total_profiles": stats["total_profiles"],
                        "date_range": stats["date_range"],
                        "geographic_coverage": stats["geographic_coverage"],
                        "parameters": ["temperature", "salinity", "pressure"]
                    }],
                    "total_found": 1,
                    "data_source": "real_argo_csv"
                }
            
            elif tool_name == "get_available_regions":
                # Get actual geographic coverage from real data
                stats = real_argo_loader.get_profile_statistics()
                geo = stats["geographic_coverage"]
                
                return {
                    "regions": [{
                        "name": "indian_ocean_region",
                        "description": "ARGO Float 1900121 coverage area",
                        "bounds": {
                            "lat_min": geo["lat_min"],
                            "lat_max": geo["lat_max"],
                            "lon_min": geo["lon_min"],
                            "lon_max": geo["lon_max"]
                        },
                        "profile_count": stats["total_profiles"],
                        "coverage": "100%"
                    }],
                    "data_source": "real_argo_csv"
                }
            
            elif tool_name == "get_available_parameters":
                return {
                    "core_parameters": [
                        {
                            "name": "temperature",
                            "unit": "degrees_Celsius",
                            "description": "Sea water temperature",
                            "available": True
                        },
                        {
                            "name": "salinity", 
                            "unit": "PSU",
                            "description": "Sea water practical salinity",
                            "available": True
                        },
                        {
                            "name": "pressure",
                            "unit": "dbar",
                            "description": "Sea water pressure",
                            "available": True
                        }
                    ],
                    "bgc_parameters": [],
                    "data_source": "real_argo_csv"
                }
            
            elif tool_name == "get_system_stats":
                # Get real statistics from the data
                stats = real_argo_loader.get_profile_statistics()
                
                return {
                    "system_status": "operational",
                    "data_statistics": stats,
                    "last_updated": datetime.now().isoformat(),
                    "data_source": "real_argo_csv"
                }
            
            else:
                return {"error": f"Unknown tool: {tool_name}"}
            
        except Exception as e:
            logger.error(f"Error processing real data for tool {tool_name}: {e}")
            # Fallback to basic response
            return {
                "error": f"Error accessing real data: {str(e)}",
                "tool": tool_name,
                "fallback": True
            }
    
    async def _send_request(self, request: Dict[str, Any]) -> MCPResponse:
        """Send request to MCP server (mock implementation)"""
        try:
            # This is a mock implementation for testing
            logger.info(f"Mock MCP request: {request.get('method', 'unknown')}")
            return MCPResponse(success=True, data={"mock": True})
            
        except Exception as e:
            logger.error(f"Error in mock MCP request: {e}")
            return MCPResponse(success=False, error=str(e))
        """Send request to MCP server"""
        try:
            if not self.server_process:
                return MCPResponse(success=False, error="MCP server not running")
            
            # Send request
            request_json = json.dumps(request) + '\n'
            self.server_process.stdin.write(request_json.encode())
            await self.server_process.stdin.drain()
            
            # Read response
            response_line = await self.server_process.stdout.readline()
            response_data = json.loads(response_line.decode().strip())
            
            if 'error' in response_data:
                return MCPResponse(
                    success=False,
                    error=response_data['error'].get('message', 'Unknown error')
                )
            
            return MCPResponse(
                success=True,
                data=response_data.get('result')
            )
            
        except Exception as e:
            logger.error(f"Error sending MCP request: {e}")
            return MCPResponse(success=False, error=str(e))
    
    async def cleanup(self):
        """Cleanup MCP resources"""
        if self.production_mode and self.server_instance:
            logger.info("Closing MCP server connection...")
            await self.server_instance.close()
            logger.info("MCP server connection closed")
        
        if self.server_process:
            self.server_process.terminate()
            await self.server_process.wait()
            logger.info("MCP server process terminated")

class MCPToolManager:
    """Manages MCP tool integration for the ARGO system"""
    
    def __init__(self, production_mode: bool = True):
        """
        Initialize MCP Tool Manager
        
        Args:
            production_mode: If True, uses real MCP server. If False, uses mock mode
        """
        self.client = ArgoMCPClient(production_mode=production_mode)
        self.tool_descriptions = {}
        self.production_mode = production_mode
    
    async def initialize(self):
        """Initialize MCP tool manager"""
        success = await self.client.initialize()
        if success:
            await self._load_tool_descriptions()
            if self.production_mode:
                logger.info("✅ MCP Tool Manager initialized in PRODUCTION mode")
            else:
                logger.info("⚠️ MCP Tool Manager initialized in MOCK mode")
        return success
    
    async def _load_tool_descriptions(self):
        """Load tool descriptions for Gemini integration"""
        tools_response = await self.client.list_tools()
        if tools_response.success:
            for tool in tools_response.data.get('tools', []):
                self.tool_descriptions[tool['name']] = {
                    'description': tool['description'],
                    'parameters': tool['parameters']
                }
    
    def get_tool_descriptions_for_gemini(self) -> str:
        """Get tool descriptions formatted for Gemini context"""
        if not self.tool_descriptions:
            return "No MCP tools available."
        
        descriptions = []
        descriptions.append("🔧 Available MCP Tools for ARGO Data Analysis:")
        
        for name, info in self.tool_descriptions.items():
            descriptions.append(f"\n• **{name}**: {info['description']}")
            
            if info['parameters']:
                descriptions.append("  Parameters:")
                for param, details in info['parameters'].items():
                    required = " (required)" if details.get('required') else ""
                    descriptions.append(f"    - {param}: {details.get('description', 'No description')}{required}")
        
        descriptions.append("\nTo use these tools, mention them in your query and I'll call them automatically.")
        return "\n".join(descriptions)
    
    async def process_query_with_tools(self, query: str) -> Dict[str, Any]:
        """Process a query using appropriate MCP tools"""
        # Determine which tools to use based on query
        tools_to_call = self._analyze_query_for_tools(query)
        
        results = {}
        for tool_name, params in tools_to_call.items():
            tool_response = await self.client.call_tool(tool_name, params)
            results[tool_name] = {
                'success': tool_response.success,
                'data': tool_response.data,
                'error': tool_response.error
            }
        
        return results
    
    def _analyze_query_for_tools(self, query: str) -> Dict[str, Dict[str, Any]]:
        """Analyze query to determine which tools to use"""
        query_lower = query.lower()
        tools = {}
        
        # Query data tool
        if any(word in query_lower for word in ['show', 'find', 'get', 'query', 'data']):
            tools['query_argo_data'] = {'query': query, 'limit': 100}
        
        # Search profiles
        if any(word in query_lower for word in ['profile', 'measurement']):
            tools['search_profiles'] = {'query': query, 'n_results': 10}
        
        # Search floats
        if any(word in query_lower for word in ['float', 'buoy', 'instrument']):
            tools['search_floats'] = {'query': query, 'n_results': 10}
        
        # Get regions
        if any(word in query_lower for word in ['region', 'area', 'ocean']):
            tools['get_available_regions'] = {}
        
        # Get parameters
        if any(word in query_lower for word in ['parameter', 'variable', 'measurement']):
            tools['get_available_parameters'] = {}
        
        # Get stats
        if any(word in query_lower for word in ['stats', 'statistics', 'status', 'info']):
            tools['get_system_stats'] = {}
        
        return tools
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.cleanup()

# Global MCP manager instance
mcp_manager = None

def initialize_mcp_manager(production_mode: bool = True):
    """Initialize global MCP manager with specified mode"""
    global mcp_manager
    mcp_manager = MCPToolManager(production_mode=production_mode)
    logger.info(f"MCP Manager created in {'PRODUCTION' if production_mode else 'MOCK'} mode")

async def initialize_mcp(production_mode: bool = True):
    """
    Initialize global MCP manager
    
    Args:
        production_mode: If True, connects to real MCP server. If False, uses mock mode
    """
    global mcp_manager
    if mcp_manager is None:
        initialize_mcp_manager(production_mode)
    return await mcp_manager.initialize()

async def get_mcp_context_for_gemini() -> str:
    """Get MCP context information for Gemini prompts"""
    if not mcp_manager.client.initialized:
        return "MCP tools not available."
    
    return mcp_manager.get_tool_descriptions_for_gemini()

async def process_query_with_mcp(query: str) -> Dict[str, Any]:
    """Process query using MCP tools"""
    return await mcp_manager.process_query_with_tools(query)