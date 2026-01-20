"""
PostgreSQL Service Manager for Windows
Handles starting/stopping PostgreSQL service with various fallback methods
"""

import subprocess
import time
import os
from typing import Tuple, List

class PostgreSQLServiceManager:
    """Manage PostgreSQL Windows service"""
    
    def __init__(self):
        self.common_service_names = [
            "postgresql-x64-13",
            "postgresql-x64-14", 
            "postgresql-x64-15",
            "postgresql-x64-16",
            "postgresql",
            "PostgreSQL"
        ]
    
    def find_postgresql_service(self) -> Tuple[bool, str]:
        """Find the actual PostgreSQL service name"""
        try:
            # Get all services and filter for postgresql
            result = subprocess.run(['sc', 'query', 'state=', 'all'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                services = result.stdout.lower()
                for service_name in self.common_service_names:
                    if service_name.lower() in services:
                        return True, service_name
            
            return False, "PostgreSQL service not found"
            
        except Exception as e:
            return False, f"Error finding service: {str(e)}"
    
    def check_service_status(self, service_name: str) -> Tuple[bool, str]:
        """Check if a specific service is running"""
        try:
            result = subprocess.run(['sc', 'query', service_name], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                output = result.stdout.upper()
                if "RUNNING" in output:
                    return True, f"Service '{service_name}' is running"
                elif "STOPPED" in output:
                    return False, f"Service '{service_name}' is stopped"
                else:
                    return False, f"Service '{service_name}' status unknown"
            else:
                return False, f"Service '{service_name}' not found"
                
        except Exception as e:
            return False, f"Error checking service: {str(e)}"
    
    def start_service(self, service_name: str) -> Tuple[bool, str]:
        """Start PostgreSQL service"""
        try:
            result = subprocess.run(['sc', 'start', service_name], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Wait for service to start
                for i in range(10):
                    time.sleep(2)
                    is_running, status = self.check_service_status(service_name)
                    if is_running:
                        return True, f"Service '{service_name}' started successfully"
                
                return False, f"Service '{service_name}' start command sent but not running"
            else:
                error_msg = result.stderr or result.stdout
                if "Access is denied" in error_msg:
                    return False, "Access denied - run as administrator to start service"
                else:
                    return False, f"Failed to start service: {error_msg}"
                    
        except Exception as e:
            return False, f"Error starting service: {str(e)}"
    
    def try_manual_start(self) -> Tuple[bool, str]:
        """Try to start PostgreSQL manually using pg_ctl"""
        try:
            # Find PostgreSQL installation
            postgres_paths = [
                r"C:\Program Files\PostgreSQL\13",
                r"C:\Program Files\PostgreSQL\14",
                r"C:\Program Files\PostgreSQL\15", 
                r"C:\Program Files\PostgreSQL\16"
            ]
            
            for base_path in postgres_paths:
                pg_ctl_path = os.path.join(base_path, "bin", "pg_ctl.exe")
                data_path = os.path.join(base_path, "data")
                
                if os.path.exists(pg_ctl_path) and os.path.exists(data_path):
                    # Try to start with pg_ctl
                    cmd = [pg_ctl_path, "start", "-D", data_path, "-l", 
                          os.path.join(data_path, "logfile")]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        return True, f"PostgreSQL started manually from {base_path}"
                    elif "already running" in result.stderr.lower():
                        return True, f"PostgreSQL already running from {base_path}"
            
            return False, "Could not start PostgreSQL manually"
            
        except Exception as e:
            return False, f"Error with manual start: {str(e)}"
    
    def comprehensive_start(self) -> Tuple[bool, str]:
        """Try all methods to start PostgreSQL"""
        
        # First, find the service
        found, service_name = self.find_postgresql_service()
        
        if found:
            # Check if already running
            is_running, status = self.check_service_status(service_name)
            if is_running:
                return True, status
            
            # Try to start service
            started, start_msg = self.start_service(service_name)
            if started:
                return True, start_msg
            else:
                print(f"Service start failed: {start_msg}")
        
        # Try manual start
        manual_started, manual_msg = self.try_manual_start()
        if manual_started:
            return True, manual_msg
        
        return False, f"All start methods failed. Last error: {manual_msg}"

def main():
    """Test service management"""
    manager = PostgreSQLServiceManager()
    
    print("PostgreSQL Service Management Test")
    print("="*50)
    
    # Find service
    found, service_name = manager.find_postgresql_service()
    print(f"Service Discovery: {service_name}")
    
    if found:
        is_running, status = manager.check_service_status(service_name)
        print(f"Service Status: {status}")
        
        if not is_running:
            print("\\nAttempting to start PostgreSQL...")
            started, start_msg = manager.comprehensive_start()
            print(f"Start Result: {start_msg}")
    else:
        print("\\nTrying manual start...")
        started, start_msg = manager.try_manual_start()
        print(f"Manual Start: {start_msg}")

if __name__ == "__main__":
    main()