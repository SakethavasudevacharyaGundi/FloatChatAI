import asyncpg
from typing import Any, Dict, List, Optional
from app.core.config import config
import logging
logger =logging.getLogger(__name__)
class DataService:

    def __init__(self):
        self.pool = None

    async def initialize(self):
        
        """
        Initialize PostgreSQL connection pool.
        """
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                database=config.DB_NAME,
                min_size=1,
                max_size=5
            )
            logger.info("PostgreSQL connection pool initialized successfully.")

    async def execute_query(
        self,
        query: str,
        *args
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results.
        """
        await self.initialize()
        
        try:
            async with self.pool.acquire() as connection:
                results = await connection.fetch(
                    query,
                    *args
                )
                return [
                    dict(record)
                    for record in results
                ]
        except Exception as e:
            logger.info(f"Database query error: {e}")
            return []

    async def execute_one(
        self,
        query: str,
        *args
    ) -> Optional[Dict[str, Any]]:
        """
        Execute query and return single record.
        """
        await self.initialize()
        
        try:
            async with self.pool.acquire() as connection:
                result = await connection.fetchrow(
                    query,
                    *args
                )
                if result:
                    return dict(result)
                return None
        except Exception as e:
            print(f"Database single query error: {e}")
            return None

    async def execute_command(
        self,
        query: str,
        *args
    ) -> bool:
        """
        Execute INSERT/UPDATE/DELETE command.
        """
        await self.initialize()
        
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(
                    query,
                    *args
                )
                return True
        except Exception as e:
            print(f"Database command error: {e}")
            return False

    async def health_check(self) -> bool:
        """
        Check database connectivity.
        """
        try:
            await self.initialize()
            
            async with self.pool.acquire() as connection:
                await connection.fetch("SELECT 1")
            return True
        except Exception as e:
            print(f"Database health check failed: {e}")
            return False

    async def close(self):
        """
        Close PostgreSQL connection pool.
        """
        if self.pool:
            await self.pool.close()
            print("PostgreSQL connection pool closed successfully.")