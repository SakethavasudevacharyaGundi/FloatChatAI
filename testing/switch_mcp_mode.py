#!/usr/bin/env python3
"""
Quick MCP Mode Switcher
Easily switch between Production and Mock modes
"""

import os
import sys

def set_mcp_mode(production: bool = True):
    """
    Set MCP mode for the project
    
    Args:
        production: True for production mode, False for mock mode
    """
    
    mode_name = "PRODUCTION" if production else "MOCK"
    
    print(f"\n{'='*60}")
    print(f"  MCP Mode Switcher")
    print(f"{'='*60}\n")
    
    # Check if .env file exists
    env_file = ".env"
    env_exists = os.path.exists(env_file)
    
    if not env_exists:
        print(f"⚠️  No .env file found. Creating one from .env.example...")
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print(f"✅ Created .env file")
        else:
            print(f"❌ No .env.example file found. Creating minimal .env...")
            with open(".env", "w") as f:
                f.write("# MCP Configuration\n")
                f.write(f"MCP_PRODUCTION_MODE={'true' if production else 'false'}\n")
                f.write("MCP_ENABLED=true\n")
            print(f"✅ Created minimal .env file")
    
    # Read current .env
    with open(env_file, "r") as f:
        lines = f.readlines()
    
    # Update MCP_PRODUCTION_MODE
    mode_set = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("MCP_PRODUCTION_MODE="):
            new_lines.append(f"MCP_PRODUCTION_MODE={'true' if production else 'false'}\n")
            mode_set = True
        else:
            new_lines.append(line)
    
    # Add if not found
    if not mode_set:
        new_lines.append(f"\n# MCP Mode Configuration\n")
        new_lines.append(f"MCP_PRODUCTION_MODE={'true' if production else 'false'}\n")
    
    # Write back
    with open(env_file, "w") as f:
        f.writelines(new_lines)
    
    print(f"✅ MCP Mode set to: {mode_name}")
    print(f"\n📋 Configuration:")
    print(f"   Mode: {mode_name}")
    print(f"   File: {env_file}")
    
    if production:
        print(f"\n⚙️  Production Mode Requirements:")
        print(f"   • PostgreSQL running on port 5433")
        print(f"   • Database: argo_ocean_data")
        print(f"   • User: postgres")
        print(f"   • ARGO data loaded")
        print(f"\n   Test with: python test_mcp_modes.py")
    else:
        print(f"\n⚙️  Mock Mode:")
        print(f"   • Uses simulated data")
        print(f"   • No database required")
        print(f"   • Good for testing")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Restart your application")
    print(f"   2. Run: python test_mcp_modes.py")
    print(f"   3. Start dashboard: streamlit run floatchat_ai_dashboard.py")
    
    print(f"\n{'='*60}\n")

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python switch_mcp_mode.py production  # Enable production mode")
        print("  python switch_mcp_mode.py mock        # Enable mock mode")
        print("\nOr just run the script and follow prompts:")
        print("  python switch_mcp_mode.py")
        print()
        
        # Interactive mode
        while True:
            choice = input("Select mode (1=Production, 2=Mock, q=Quit): ").strip().lower()
            
            if choice == 'q':
                print("Cancelled.")
                return
            elif choice == '1' or choice == 'production':
                set_mcp_mode(production=True)
                break
            elif choice == '2' or choice == 'mock':
                set_mcp_mode(production=False)
                break
            else:
                print("Invalid choice. Please enter 1, 2, or q")
    else:
        mode = sys.argv[1].lower()
        
        if mode in ['production', 'prod', 'p', 'true', '1']:
            set_mcp_mode(production=True)
        elif mode in ['mock', 'm', 'false', '0']:
            set_mcp_mode(production=False)
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Use: production or mock")
            sys.exit(1)

if __name__ == "__main__":
    main()
