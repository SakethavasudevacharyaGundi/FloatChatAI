"""
PostgreSQL Configuration Tester and Auto-Configurator
Tests different connection parameters and configurations
"""

import os
import psycopg2
import subprocess
from typing import Dict, List, Tuple
import json

class PostgreSQLConfigTester:
    """Test and configure PostgreSQL connection parameters"""
    
    def __init__(self):
        self.test_configs = [
            # Standard configurations
            {"host": "localhost", "port": "5432", "user": "postgres", "password": "password"},
            {"host": "localhost", "port": "5432", "user": "postgres", "password": "postgres"},
            {"host": "localhost", "port": "5432", "user": "postgres", "password": ""},
            {"host": "127.0.0.1", "port": "5432", "user": "postgres", "password": "password"},
            {"host": "127.0.0.1", "port": "5432", "user": "postgres", "password": "postgres"},
            
            # Port 5433 (found in config)
            {"host": "localhost", "port": "5433", "user": "postgres", "password": "password"},
            {"host": "localhost", "port": "5433", "user": "postgres", "password": "postgres"},
            {"host": "localhost", "port": "5433", "user": "postgres", "password": ""},
            {"host": "localhost", "port": "5433", "user": "postgres", "password": "admin"},
            {"host": "localhost", "port": "5433", "user": "postgres", "password": "123456"},
            {"host": "localhost", "port": "5433", "user": "postgres", "password": "root"},
            {"host": "127.0.0.1", "port": "5433", "user": "postgres", "password": "password"},
            {"host": "127.0.0.1", "port": "5433", "user": "postgres", "password": "postgres"},
            
            # Different ports
            {"host": "localhost", "port": "5434", "user": "postgres", "password": "password"},
        ]
        
    def test_connection(self, config: Dict[str, str]) -> Tuple[bool, str]:
        """Test a specific configuration"""
        try:
            conn = psycopg2.connect(
                host=config['host'],
                port=config['port'],
                user=config['user'],
                password=config['password'],
                database='postgres',
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return True, f"SUCCESS: {version}"
            
        except psycopg2.OperationalError as e:
            return False, f"Connection failed: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def find_working_config(self) -> Tuple[bool, Dict[str, str], str]:
        """Find a working PostgreSQL configuration"""
        
        for i, config in enumerate(self.test_configs):
            print(f"Testing config {i+1}/{len(self.test_configs)}: "
                  f"{config['user']}@{config['host']}:{config['port']}")
            
            success, message = self.test_connection(config)
            
            if success:
                print(f"✓ FOUND WORKING CONFIG: {message}")
                return True, config, message
            else:
                print(f"✗ Failed: {message}")
        
        return False, {}, "No working configuration found"
    
    def get_postgresql_info(self) -> Dict:
        """Get comprehensive PostgreSQL installation info"""
        info = {
            "installation_paths": [],
            "service_status": {},
            "config_files": [],
            "data_directories": []
        }
        
        # Check installation paths
        base_paths = [
            r"C:\Program Files\PostgreSQL",
            r"C:\PostgreSQL"
        ]
        
        for base_path in base_paths:
            if os.path.exists(base_path):
                for version_dir in os.listdir(base_path):
                    version_path = os.path.join(base_path, version_dir)
                    if os.path.isdir(version_path):
                        bin_path = os.path.join(version_path, "bin")
                        data_path = os.path.join(version_path, "data")
                        
                        info["installation_paths"].append({
                            "version": version_dir,
                            "path": version_path,
                            "bin_exists": os.path.exists(bin_path),
                            "data_exists": os.path.exists(data_path)
                        })
                        
                        # Check for config files
                        config_files = ["postgresql.conf", "pg_hba.conf"]
                        for config_file in config_files:
                            config_path = os.path.join(data_path, config_file)
                            if os.path.exists(config_path):
                                info["config_files"].append({
                                    "file": config_file,
                                    "path": config_path,
                                    "version": version_dir
                                })
        
        return info
    
    def check_postgresql_conf(self, config_path: str) -> Dict:
        """Check PostgreSQL configuration file"""
        try:
            with open(config_path, 'r') as f:
                lines = f.readlines()
            
            settings = {}
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        settings[key.strip()] = value.strip()
            
            return {
                "file_path": config_path,
                "settings": settings,
                "key_settings": {
                    "port": settings.get("port", "not found"),
                    "listen_addresses": settings.get("listen_addresses", "not found"),
                    "max_connections": settings.get("max_connections", "not found")
                }
            }
            
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main testing function"""
    tester = PostgreSQLConfigTester()
    
    print("="*70)
    print("POSTGRESQL CONFIGURATION TESTER")
    print("="*70)
    
    # Get installation info
    print("\\n1. PostgreSQL Installation Analysis:")
    print("-" * 50)
    info = tester.get_postgresql_info()
    
    print(f"Found {len(info['installation_paths'])} PostgreSQL installations:")
    for install in info['installation_paths']:
        print(f"  Version {install['version']}: {install['path']}")
        print(f"    Bin directory: {'✓' if install['bin_exists'] else '✗'}")
        print(f"    Data directory: {'✓' if install['data_exists'] else '✗'}")
    
    print(f"\\nFound {len(info['config_files'])} configuration files:")
    for config in info['config_files']:
        print(f"  {config['file']} (v{config['version']}): {config['path']}")
    
    # Check configuration files
    if info['config_files']:
        print("\\n2. Configuration File Analysis:")
        print("-" * 50)
        for config in info['config_files']:
            if config['file'] == 'postgresql.conf':
                conf_analysis = tester.check_postgresql_conf(config['path'])
                if 'error' not in conf_analysis:
                    print(f"  {config['file']} settings:")
                    for key, value in conf_analysis['key_settings'].items():
                        print(f"    {key}: {value}")
                else:
                    print(f"  Error reading {config['file']}: {conf_analysis['error']}")
    
    # Test connections
    print("\\n3. Connection Testing:")
    print("-" * 50)
    
    success, working_config, message = tester.find_working_config()
    
    if success:
        print(f"\\n✓ SUCCESS! Working configuration found:")
        print(f"  Host: {working_config['host']}")
        print(f"  Port: {working_config['port']}")
        print(f"  User: {working_config['user']}")
        print(f"  Password: {'***' if working_config['password'] else '(empty)'}")
        print(f"  Database: postgres")
        
        # Save config
        config_file = os.path.join(os.path.dirname(__file__), "postgresql_config.json")
        with open(config_file, 'w') as f:
            json.dump(working_config, f, indent=2)
        print(f"\\nConfiguration saved to: {config_file}")
        
        return True
    else:
        print(f"\\n✗ FAILED: {message}")
        print("\\nTroubleshooting suggestions:")
        print("1. Check if PostgreSQL service is running")
        print("2. Verify password (common defaults: 'password', 'postgres', or empty)")
        print("3. Check firewall settings")
        print("4. Verify pg_hba.conf allows local connections")
        
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)