#!/usr/bin/env python3
"""
Test MCP Client in both Production and Mock modes
Run this to verify your MCP setup is working correctly
"""

import asyncio
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config

async def test_mcp_mode(production_mode: bool):
    """Test MCP in specified mode"""
    
    mode_name = "PRODUCTION" if production_mode else "MOCK"
    print(f"\n{'='*60}")
    print(f"Testing MCP in {mode_name} Mode")
    print(f"{'='*60}\n")
    
    try:
        from mcp.mcp_client import ArgoMCPClient
        
        # Initialize client
        client = ArgoMCPClient(production_mode=production_mode)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing MCP client...")
        success = await client.initialize()
        
        if not success:
            print(f"❌ Failed to initialize MCP client in {mode_name} mode")
            return False
        
        print(f"✅ MCP client initialized successfully")
        print(f"📊 Mode: {mode_name}")
        print(f"🔧 Available tools: {len(client.available_tools)}")
        
        # List all tools
        print(f"\n📋 Available MCP Tools:")
        for i, (tool_name, tool_info) in enumerate(client.available_tools.items(), 1):
            print(f"  {i:2d}. {tool_name}")
            print(f"      {tool_info.get('description', 'No description')}")
        
        # Test a simple tool call
        print(f"\n🧪 Testing tool execution...")
        
        test_cases = [
            ("get_system_stats", {}, "System Statistics"),
            ("get_available_parameters", {}, "Available Parameters"),
        ]
        
        for tool_name, params, description in test_cases:
            if tool_name in client.available_tools:
                print(f"\n--- Testing: {description} ---")
                result = await client.call_tool(tool_name, params)
                
                if result.success:
                    print(f"✅ {tool_name}: Success")
                    if result.data:
                        # Print first few keys of result
                        data_keys = list(result.data.keys())[:3] if isinstance(result.data, dict) else []
                        if data_keys:
                            print(f"   Data keys: {', '.join(data_keys)}")
                else:
                    print(f"❌ {tool_name}: Failed - {result.error}")
        
        # Cleanup
        await client.cleanup()
        print(f"\n✅ {mode_name} mode test completed successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run tests for both modes"""
    
    print("\n" + "="*60)
    print("MCP Client Mode Testing Suite")
    print("="*60)
    print(f"Configuration:")
    print(f"  MCP Enabled: {config.MCP_ENABLED}")
    print(f"  Default Mode: {'PRODUCTION' if config.MCP_PRODUCTION_MODE else 'MOCK'}")
    print(f"  PostgreSQL: {config.MCP_DB_HOST}:{config.MCP_DB_PORT}/{config.MCP_DB_NAME}")
    
    results = {}
    
    # Test Mock Mode
    print("\n\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  TEST 1: MOCK MODE (Simulated Data)".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    results['mock'] = await test_mcp_mode(production_mode=False)
    
    # Test Production Mode
    print("\n\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  TEST 2: PRODUCTION MODE (Real MCP Server)".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    results['production'] = await test_mcp_mode(production_mode=True)
    
    # Summary
    print("\n\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Mock Mode:       {'✅ PASSED' if results['mock'] else '❌ FAILED'}")
    print(f"Production Mode: {'✅ PASSED' if results['production'] else '❌ FAILED'}")
    print("="*60)
    
    if results['production']:
        print("\n🎉 SUCCESS! Your MCP is configured for production mode!")
        print("   You can now use MCP_PRODUCTION_MODE=true in your config")
    else:
        print("\n⚠️  Production mode failed. Check:")
        print("   1. PostgreSQL is running on port 5433")
        print("   2. Database 'argo_ocean_data' exists")
        print("   3. Database credentials are correct")
        print("   4. ARGO data has been loaded into the database")
    
    print("\n💡 To switch modes, set MCP_PRODUCTION_MODE environment variable:")
    print("   - export MCP_PRODUCTION_MODE=true  # Use real MCP server")
    print("   - export MCP_PRODUCTION_MODE=false # Use mock data")

if __name__ == "__main__":
    asyncio.run(main())
