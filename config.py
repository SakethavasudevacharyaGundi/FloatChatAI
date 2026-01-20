"""
Configuration file for FloatChatAI - ARGO Ocean Data Explorer
This is a template - copy this file and fill in your actual values
"""

import os
from pathlib import Path

class Config:
    """Main configuration class"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
    
    # Database configuration (loaded from environment variables)
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', '5433'))
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'your-database-password-here')
    DB_NAME = os.getenv('DB_NAME', 'argo_ocean_data')
    
    # Google Gemini API configuration
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'your-google-api-key-here')
    GOOGLE_API_KEY_PRIMARY = os.getenv('GOOGLE_API_KEY_PRIMARY', 'your-primary-google-api-key-here')
    GOOGLE_API_KEY_FALLBACK = os.getenv('GOOGLE_API_KEY_FALLBACK', 'your-fallback-google-api-key-here')
    
    # LLM Configuration
    LLM_MODEL = "gemini-1.5-pro"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 8192
    
    # RAG Configuration
    VECTOR_DB_TYPE = "chroma"
    EMBEDDING_MODEL = "models/embedding-001"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # MCP Configuration
    MCP_ENABLED = True
    MCP_SERVER_MODE = "local"  # or "remote"
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_RELOAD = True
    
    # Streamlit Configuration
    STREAMLIT_PORT = 8501
    STREAMLIT_THEME = "dark"
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Data processing
    MAX_PROFILES_PER_QUERY = 1000
    CACHE_ENABLED = True
    CACHE_TTL = 3600  # seconds

# Create a global config instance
config = Config()
