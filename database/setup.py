"""
PostgreSQL Database Setup and Management Module
Comprehensive solution for ARGO Ocean Data Explorer database setup.
No compromises - handles all edge cases and provides clear error reporting.
"""

import os
import sys
import subprocess
import psycopg2
import asyncpg
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostgreSQLSetup:
    """Comprehensive PostgreSQL setup and management class"""
    
    def __init__(self, config: Dict[str, str] = None):
        """Initialize with database configuration"""
        self.config = config or {
            "host": os.getenv('DB_HOST', 'localhost'),
            "port": os.getenv('DB_PORT', '5433'),
            "user": os.getenv('DB_USER', 'postgres'),
            "password": os.getenv('DB_PASSWORD', 'your-database-password-here'),
            "database": os.getenv('DB_NAME', 'argo_ocean_data')
        }
        self.connection_url = f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        self.admin_url = f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/postgres"
        
    def check_postgresql_installation(self) -> Tuple[bool, str]:
        """
        Check if PostgreSQL is installed and accessible
        Returns: (is_installed, status_message)
        """
        try:
            # Check common PostgreSQL installation paths on Windows
            common_paths = [
                r"C:\Program Files\PostgreSQL\13\bin",
                r"C:\Program Files\PostgreSQL\14\bin", 
                r"C:\Program Files\PostgreSQL\15\bin",
                r"C:\Program Files\PostgreSQL\16\bin",
                r"C:\PostgreSQL\13\bin",
                r"C:\PostgreSQL\14\bin",
                r"C:\PostgreSQL\15\bin",
                r"C:\PostgreSQL\16\bin"
            ]
            
            psql_path = None
            for path in common_paths:
                potential_psql = os.path.join(path, "psql.exe")
                if os.path.exists(potential_psql):
                    psql_path = potential_psql
                    # Add to PATH for this session
                    if path not in os.environ.get('PATH', ''):
                        os.environ['PATH'] = f"{path};{os.environ.get('PATH', '')}"
                    break
            
            if psql_path:
                # Test with full path
                result = subprocess.run([psql_path, '--version'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    logger.info(f"PostgreSQL found: {version}")
                    return True, f"PostgreSQL installed: {version}"
            
            # Try psql command from PATH
            result = subprocess.run(['psql', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"PostgreSQL found: {version}")
                return True, f"PostgreSQL installed: {version}"
            else:
                return False, "PostgreSQL command not found"
                
        except subprocess.TimeoutExpired:
            return False, "PostgreSQL command timed out"
        except FileNotFoundError:
            return False, "PostgreSQL not installed - psql command not found"
        except Exception as e:
            return False, f"Error checking PostgreSQL: {str(e)}"
    
    def check_postgresql_service(self) -> Tuple[bool, str]:
        """
        Check if PostgreSQL service is running
        Returns: (is_running, status_message)
        """
        try:
            # Try to connect to default postgres database
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database='postgres',
                connect_timeout=5
            )
            conn.close()
            return True, "PostgreSQL service is running"
        except psycopg2.OperationalError as e:
            error_msg = str(e)
            if "password authentication failed" in error_msg:
                return False, f"Authentication failed - check username/password: {error_msg}"
            elif "could not connect to server" in error_msg:
                return False, f"Cannot connect to PostgreSQL server - check if service is running: {error_msg}"
            elif "database" in error_msg and "does not exist" in error_msg:
                return True, "PostgreSQL service running but database doesn't exist (this is expected)"
            else:
                return False, f"PostgreSQL connection error: {error_msg}"
        except Exception as e:
            return False, f"Unexpected error connecting to PostgreSQL: {str(e)}"
    
    def create_database(self) -> Tuple[bool, str]:
        """
        Create the ARGO ocean data database
        Returns: (success, message)
        """
        try:
            # Connect to postgres database to create new database
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database='postgres'
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if database already exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.config['database'],))
            if cursor.fetchone():
                cursor.close()
                conn.close()
                return True, f"Database '{self.config['database']}' already exists"
            
            # Create database
            cursor.execute(f'CREATE DATABASE "{self.config["database"]}"')
            cursor.close()
            conn.close()
            
            logger.info(f"Created database: {self.config['database']}")
            return True, f"Successfully created database '{self.config['database']}'"
            
        except psycopg2.Error as e:
            return False, f"PostgreSQL error creating database: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error creating database: {str(e)}"
    
    def execute_schema(self, schema_path: str) -> Tuple[bool, str]:
        """
        Execute schema SQL file to create tables
        Returns: (success, message)
        """
        try:
            schema_file = Path(schema_path)
            if not schema_file.exists():
                return False, f"Schema file not found: {schema_path}"
            
            # Read schema file
            schema_sql = schema_file.read_text(encoding='utf-8')
            
            # Connect to target database
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database']
            )
            cursor = conn.cursor()
            
            # Execute schema
            cursor.execute(schema_sql)
            conn.commit()
            
            # Verify tables were created
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            if not tables:
                return False, "Schema executed but no tables were created"
            
            logger.info(f"Schema executed successfully. Created tables: {', '.join(tables)}")
            return True, f"Schema executed successfully. Created tables: {', '.join(tables)}"
            
        except psycopg2.Error as e:
            return False, f"PostgreSQL error executing schema: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error executing schema: {str(e)}"
    
    async def test_async_connection(self) -> Tuple[bool, str]:
        """
        Test async connection using asyncpg
        Returns: (success, message)
        """
        try:
            conn = await asyncpg.connect(self.connection_url)
            
            # Test basic query
            result = await conn.fetchval("SELECT version()")
            await conn.close()
            
            return True, f"Async connection successful. PostgreSQL version: {result}"
            
        except asyncpg.InvalidCatalogNameError:
            return False, f"Database '{self.config['database']}' does not exist"
        except asyncpg.InvalidPasswordError:
            return False, "Invalid username or password"
        except asyncpg.CannotConnectNowError:
            return False, "Cannot connect to PostgreSQL server - check if service is running"
        except Exception as e:
            return False, f"Async connection error: {str(e)}"
    
    def get_database_info(self) -> Dict[str, any]:
        """
        Get comprehensive database information
        Returns: Dictionary with database details
        """
        try:
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database']
            )
            cursor = conn.cursor()
            
            info = {}
            
            # Get PostgreSQL version
            cursor.execute("SELECT version()")
            info['postgresql_version'] = cursor.fetchone()[0]
            
            # Get database size
            cursor.execute("SELECT pg_size_pretty(pg_database_size(%s))", (self.config['database'],))
            info['database_size'] = cursor.fetchone()[0]
            
            # Get table information
            cursor.execute("""
                SELECT 
                    table_name,
                    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
                FROM information_schema.tables t
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = cursor.fetchall()
            info['tables'] = [{'name': table[0], 'columns': table[1]} for table in tables]
            
            # Get total record counts for each table
            for table_info in info['tables']:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_info['name']}")
                    table_info['record_count'] = cursor.fetchone()[0]
                except:
                    table_info['record_count'] = 'Error'
            
            cursor.close()
            conn.close()
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def run_comprehensive_check(self) -> Dict[str, any]:
        """
        Run comprehensive PostgreSQL setup check
        Returns: Dictionary with all check results
        """
        results = {
            'timestamp': str(asyncio.get_event_loop().time()),
            'config': self.config.copy(),
            'checks': {}
        }
        
        # Remove password from results for security
        results['config']['password'] = '***'
        
        # Check installation
        is_installed, install_msg = self.check_postgresql_installation()
        results['checks']['installation'] = {'success': is_installed, 'message': install_msg}
        
        if not is_installed:
            results['checks']['overall_status'] = {'success': False, 'message': 'PostgreSQL not installed'}
            return results
        
        # Check service
        is_running, service_msg = self.check_postgresql_service()
        results['checks']['service'] = {'success': is_running, 'message': service_msg}
        
        if not is_running:
            results['checks']['overall_status'] = {'success': False, 'message': 'PostgreSQL service not running'}
            return results
        
        # Create database
        db_created, db_msg = self.create_database()
        results['checks']['database_creation'] = {'success': db_created, 'message': db_msg}
        
        if not db_created:
            results['checks']['overall_status'] = {'success': False, 'message': 'Failed to create database'}
            return results
        
        # Execute schema
        schema_path = Path(__file__).parent / 'postgresql_schema.sql'
        schema_executed, schema_msg = self.execute_schema(str(schema_path))
        results['checks']['schema_execution'] = {'success': schema_executed, 'message': schema_msg}
        
        if not schema_executed:
            results['checks']['overall_status'] = {'success': False, 'message': 'Failed to execute schema'}
            return results
        
        # Test async connection
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async_success, async_msg = loop.run_until_complete(self.test_async_connection())
        results['checks']['async_connection'] = {'success': async_success, 'message': async_msg}
        
        if not async_success:
            results['checks']['overall_status'] = {'success': False, 'message': 'Async connection failed'}
            return results
        
        # Get database info
        db_info = self.get_database_info()
        results['database_info'] = db_info
        
        # Overall success
        results['checks']['overall_status'] = {'success': True, 'message': 'PostgreSQL setup completed successfully'}
        
        return results

def main():
    """Main function for testing setup"""
    setup = PostgreSQLSetup()
    
    print("="*60)
    print("ARGO OCEAN DATA EXPLORER - POSTGRESQL SETUP")
    print("="*60)
    
    results = setup.run_comprehensive_check()
    
    print(f"\nTimestamp: {results['timestamp']}")
    print(f"Configuration: {results['config']}")
    
    print("\nSetup Results:")
    print("-" * 40)
    
    for check_name, check_result in results['checks'].items():
        status = "✓ PASS" if check_result['success'] else "✗ FAIL"
        print(f"{check_name.replace('_', ' ').title()}: {status}")
        print(f"  Message: {check_result['message']}")
    
    if 'database_info' in results:
        print("\nDatabase Information:")
        print("-" * 40)
        info = results['database_info']
        if 'error' not in info:
            print(f"PostgreSQL Version: {info['postgresql_version']}")
            print(f"Database Size: {info['database_size']}")
            print(f"Tables: {len(info['tables'])}")
            for table in info['tables']:
                print(f"  - {table['name']}: {table['columns']} columns, {table['record_count']} records")
        else:
            print(f"Error getting database info: {info['error']}")
    
    overall_success = results['checks']['overall_status']['success']
    print(f"\nOverall Setup Status: {'✓ SUCCESS' if overall_success else '✗ FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)