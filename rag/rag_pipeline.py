#!/usr/bin/env python3
"""
Enhanced ARGO Ocean Data RAG Pipeline System

Comprehensive Retrieval-Augmented Generation (RAG) system for ARGO oceanographic data.
Integrates PostgreSQL database, vector embeddings, semantic search, and enhanced context
generation for sophisticated oceanographic data analysis and natural language querying.

Features:
- PostgreSQL database integration with real ARGO data
- Vector database with ChromaDB for semantic search
- Advanced embedding models for oceanographic content
- Multi-modal document processing (profiles, metadata, patterns)
- Contextual search with domain-specific knowledge
- Dynamic context augmentation for LLM queries
- Temporal and spatial filtering capabilities
"""

import asyncio
import asyncpg
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
try:
    from langchain_community.vectorstores import Chroma
except Exception:  # fallback for older installs
    from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    from langchain.embeddings import HuggingFaceEmbeddings
import json
import re
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import hashlib
import pickle
import os

# Import existing modules
from vector_db.vector_store import ArgoVectorStore
from rag.query_processor import ArgoQueryProcessor
from models.data_models import QueryRequest, QueryResponse
from config import config

# Vector database and embedding imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available, using fallback vector storage")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available, using fallback embeddings")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArgoRAGPipeline:
    """
    Enhanced RAG Pipeline for ARGO Ocean Data with PostgreSQL Integration
    
    Features:
    - PostgreSQL database integration with real ARGO data (port 5433)
    - Vector database with ChromaDB for semantic search
    - Advanced embedding models for oceanographic content
    - Multi-modal document processing (profiles, metadata, patterns)
    - Contextual search with domain-specific knowledge
    - Dynamic context augmentation for LLM queries
    - Temporal and spatial filtering capabilities
    - Enhanced natural language to SQL conversion
    """
    
    def __init__(self, use_enhanced_features: bool = True):
        """Initialize enhanced RAG pipeline with PostgreSQL integration"""
        
        # Legacy components (keep for compatibility)
        self.vector_store = ArgoVectorStore(config.CHROMA_PERSIST_DIRECTORY)
        self.query_processor = ArgoQueryProcessor()
        
        # PostgreSQL configuration (real database)
        self.db_config = {
            'host': 'localhost',
            'port': 5433,
            'database': 'argo_ocean_data',
            'user': 'postgres',
            'password': '02082006'
        }
        self.db_pool = None
        
        # Enhanced RAG configuration
        self.use_enhanced_features = use_enhanced_features
        self.enhanced_vector_db_path = str(Path.cwd() / "vector_db" / "enhanced_argo_rag")
        Path(self.enhanced_vector_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Vector database setup
        self.chroma_client = None
        self.enhanced_collection = None
        
        # Document cache for enhanced features
        self.document_cache = {}
        self.cache_file = Path(self.enhanced_vector_db_path).parent / "enhanced_document_cache.pkl"
        
        # RAG configuration
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.max_context_length = 4000
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Initialize embeddings and LLM (use Google Gemini if available)
        self._initialize_models()
        
        # Initialize enhanced vector database
        if self.use_enhanced_features:
            self._initialize_enhanced_vector_db()
        
        # Legacy components
        self.chroma_db = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.schema_info = self._get_schema_info()
        self.nl_to_sql_prompt = self._create_nl_to_sql_prompt()
        
        logger.info("Enhanced ARGO RAG Pipeline initialized")
    
    @property
    def documents(self):
        """Get documents from the document cache"""
        if hasattr(self, 'document_cache') and self.document_cache:
            return list(self.document_cache.values())
        return []
    
    def _initialize_models(self):
        """Initialize LLM and embedding models"""
        try:
            if config.GEMINI_API_KEY:
                # Configure Google Gemini API
                genai.configure(api_key=config.GEMINI_API_KEY)
                
                # Initialize Gemini LLM
                self.llm = ChatGoogleGenerativeAI(
                    model=config.GEMINI_MODEL,
                    temperature=0.1,
                    max_output_tokens=1000,
                    google_api_key=config.GEMINI_API_KEY
                )
                
                # Initialize Gemini embeddings
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=config.GEMINI_EMBEDDING_MODEL,
                    google_api_key=config.GEMINI_API_KEY
                )
                
                # Also initialize SentenceTransformers for enhanced features
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.enhanced_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                    logger.info("Using Google Gemini + SentenceTransformers for enhanced RAG...")
                else:
                    self.enhanced_embedding_model = None
                    logger.info("Using Google Gemini for LLM and embeddings...")
            else:
                raise ValueError("No Google API key provided")
        except Exception as e:
            # Fallback to local models
            logger.warning(f"Error initializing Gemini (falling back to local models): {e}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.llm = None  # Will use rule-based approach
            
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.enhanced_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                self.enhanced_embedding_model = None
    
    def _initialize_enhanced_vector_db(self):
        """Initialize enhanced ChromaDB vector database"""
        
        if CHROMA_AVAILABLE:
            try:
                # Initialize ChromaDB client
                self.chroma_client = chromadb.PersistentClient(
                    path=self.enhanced_vector_db_path,
                    settings=Settings(anonymized_telemetry=False)
                )
                
                # Get or create enhanced collection
                collection_name = "enhanced_argo_oceanographic_data"
                try:
                    self.enhanced_collection = self.chroma_client.get_collection(
                        name=collection_name,
                        embedding_function=None  # We'll handle embeddings manually
                    )
                    logger.info(f"Loaded existing enhanced collection '{collection_name}'")
                except:
                    self.enhanced_collection = self.chroma_client.create_collection(
                        name=collection_name,
                        embedding_function=None,
                        metadata={"description": "Enhanced ARGO oceanographic data embeddings with PostgreSQL integration"}
                    )
                    logger.info(f"Created new enhanced collection '{collection_name}'")
                
                # Get collection stats
                count = self.enhanced_collection.count()
                logger.info(f"Enhanced vector database initialized with {count} documents")
                
                # Load document cache
                self._load_document_cache()
                
            except Exception as e:
                logger.error(f"Enhanced ChromaDB initialization failed: {str(e)}")
                self.chroma_client = None
                self.enhanced_collection = None
        else:
            logger.warning("ChromaDB not available for enhanced features")
    
    async def initialize_postgresql(self) -> bool:
        """Initialize PostgreSQL connection pool"""
        
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
            logger.info("PostgreSQL connection pool initialized for enhanced RAG")
            return True
            
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {str(e)}")
            return False
    
    def _load_document_cache(self):
        """Load document cache from disk"""
        
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.document_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.document_cache)} enhanced documents from cache")
            except Exception as e:
                logger.warning(f"Failed to load enhanced document cache: {str(e)}")
                self.document_cache = {}
        else:
            self.document_cache = {}
    
    async def search_enhanced_rag(self, query: str, top_k: int = 5, similarity_threshold: float = 0.5) -> Dict[str, Any]:
        """Search enhanced RAG system for relevant documents"""
        
        try:
            if not self.enhanced_collection:
                return {"error": "Enhanced RAG collection not available"}
            
            # Generate query embedding
            query_embedding = self._generate_enhanced_embedding(query)
            
            # Search vector database
            results = self.enhanced_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    similarity = 1 - results['distances'][0][i]  # Convert distance to similarity
                    
                    # Filter by similarity threshold
                    if similarity >= similarity_threshold:
                        doc_id = results['ids'][0][i]
                        doc = {
                            'id': doc_id,
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'document_type': results['metadatas'][0][i].get('document_type', 'unknown')
                        }
                        
                        result = {
                            'document': doc,
                            'similarity_score': similarity,
                            'relevance_explanation': f"Document matches query with {similarity:.3f} similarity based on semantic embeddings"
                        }
                        formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Enhanced RAG search failed: {str(e)}")
            return {"error": f"Search failed: {str(e)}"}
    
    async def get_enhanced_context_for_llm(self, query: str, top_k: int = 3, max_context_tokens: int = 2000) -> str:
        """Get enhanced context from RAG system for LLM processing"""
        
        try:
            # Search for relevant documents with lower threshold for context
            results = await self.search_enhanced_rag(query, top_k=top_k, similarity_threshold=0.3)
            
            if not results:
                return f"No relevant documents found for query: {query}"
            
            # Build context string for LLM
            context_parts = [f"Enhanced ARGO Oceanographic Data Context for query: '{query}'\n"]
            
            for i, result in enumerate(results, 1):
                doc = result['document']
                similarity = result['similarity_score']
                
                context_parts.append(f"\n--- Document {i} ({doc['document_type']}) - Similarity: {similarity:.3f} ---")
                context_parts.append(f"Content: {doc['content'][:500]}...")
                
                # Add key metadata
                metadata = doc['metadata']
                if 'region' in metadata:
                    context_parts.append(f"Region: {metadata['region']}")
                if 'datetime' in metadata:
                    context_parts.append(f"Date: {metadata['datetime']}")
                
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Enhanced context generation failed: {str(e)}")
            return f"Error generating enhanced context: {str(e)}"
    
    
    
    def _save_document_cache(self):
        """Save document cache to disk"""
        
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.document_cache, f)
            logger.info(f"Saved {len(self.document_cache)} enhanced documents to cache")
        except Exception as e:
            logger.warning(f"Failed to save enhanced document cache: {str(e)}")
    
    def _generate_enhanced_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using enhanced model"""
        
        if self.enhanced_embedding_model:
            try:
                embedding = self.enhanced_embedding_model.encode(text)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Enhanced embedding generation failed: {str(e)}")
        
        # Fallback: simple hash-based embedding
        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, min(len(text_hash), self.embedding_dimension * 2), 2)]
        
        # Pad or truncate to correct dimension
        if len(embedding) < self.embedding_dimension:
            embedding.extend([0.0] * (self.embedding_dimension - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dimension]
        
        return embedding
        
    def _get_schema_info(self) -> str:
        """Get PostgreSQL database schema information for SQL generation"""
        return """
        PostgreSQL Database Schema for ARGO Oceanographic Data:
        
        Table: floats
        Columns:
        - float_id (INTEGER): Primary key, unique identifier for each ARGO float
        - model (TEXT): Float model type
        - platform_number (TEXT): Platform number identifier
        - deploy_date (DATE): Date when float was deployed
        - last_lat (REAL): Last known latitude coordinate (-90 to 90)
        - last_lon (REAL): Last known longitude coordinate (-180 to 180)
        - region (TEXT): Ocean region (Indian Ocean, Atlantic Ocean, Pacific Ocean, etc.)
        - status (TEXT): Float status (active, inactive, lost)
        - total_profiles (INTEGER): Total number of profiles collected
        - last_profile_date (DATE): Date of last profile measurement
        - created_at (TIMESTAMP): Record creation timestamp
        - updated_at (TIMESTAMP): Record update timestamp
        
        Table: profiles
        Columns:
        - profile_id (INTEGER): Primary key, unique identifier for each profile
        - float_id (INTEGER): Foreign key referencing floats.float_id
        - cycle_number (INTEGER): Cycle number for this profile
        - profile_datetime (TIMESTAMP): Date and time of profile measurement
        - lat (REAL): Latitude coordinate for this profile (-90 to 90)
        - lon (REAL): Longitude coordinate for this profile (-180 to 180)
        - qc_status (TEXT): Quality control status
        - created_at (TIMESTAMP): Record creation timestamp
        - updated_at (TIMESTAMP): Record update timestamp
        
        Table: measurements
        Columns:
        - meas_id (INTEGER): Primary key, unique identifier for each measurement
        - profile_id (INTEGER): Foreign key referencing profiles.profile_id
        - depth_m (REAL): Depth in meters (0 to 6000)
        - temperature_c (REAL): Temperature in Celsius
        - salinity_psu (REAL): Salinity in PSU (Practical Salinity Units)
        - pressure_dbar (REAL): Pressure in decibars
        - created_at (TIMESTAMP): Record creation timestamp
        
        Table: profile_summaries (automatically generated)
        Columns:
        - profile_id (INTEGER): Primary key, references profiles.profile_id
        - measurement_count (INTEGER): Number of measurements in profile
        - min_depth (REAL): Minimum depth in profile
        - max_depth (REAL): Maximum depth in profile
        - avg_temperature (REAL): Average temperature in profile
        - avg_salinity (REAL): Average salinity in profile
        - created_at (TIMESTAMP): Record creation timestamp
        - updated_at (TIMESTAMP): Record update timestamp
        
        Common query patterns:
        1. Temperature/Salinity profiles: SELECT p.*, m.* FROM profiles p JOIN measurements m ON p.profile_id = m.profile_id WHERE ...
        2. Float information: SELECT * FROM floats WHERE ...
        3. Spatial queries: Use lat/lon with BETWEEN or comparison operators
        4. Temporal queries: Use profile_datetime with DATE() function or EXTRACT()
        5. Regional queries: Use region column in floats table
        6. Parameter filtering: Use temperature_c, salinity_psu, pressure_dbar columns
        7. Depth-based queries: Use depth_m column with range filters
        8. Profile summaries: SELECT p.*, ps.* FROM profiles p JOIN profile_summaries ps ON p.profile_id = ps.profile_id WHERE ...
        
        Current Database Status:
        - Real ARGO data from Float 1900121 (Indian Ocean)
        - 99 profiles with 4,380 measurements
        - Operational period: 2002-2005
        - Geographic coverage: Indian Ocean region
        """
    
    def _create_nl_to_sql_prompt(self) -> PromptTemplate:
        """Create prompt template for natural language to SQL conversion"""
        template = """
        You are an expert at converting natural language questions about oceanographic data into SQL queries.
        
        {schema_info}
        
        Rules for SQL generation:
        1. Always use proper table names (argo_profiles, argo_floats)
        2. For date queries, use DATE() function: DATE(timestamp) = '2023-03-15'
        3. For coordinate queries, use BETWEEN: latitude BETWEEN -10 AND 10
        4. For recent data, use datetime comparisons: timestamp > datetime('now', '-30 days')
        5. Always include LIMIT to prevent large result sets (default 100)
        6. Use appropriate JOINs when combining float and profile data
        7. For visualization queries, include necessary columns for plotting
        8. Return only the SQL query, no explanations
        
        Examples:
        
        Question: "Show me salinity profiles near the equator in March 2023"
        SQL: SELECT profile_id, latitude, longitude, depth, salinity, timestamp FROM argo_profiles WHERE latitude BETWEEN -5 AND 5 AND DATE(timestamp) LIKE '2023-03%' AND salinity IS NOT NULL ORDER BY timestamp LIMIT 100;
        
        Question: "Find active ARGO floats in the Indian Ocean"
        SQL: SELECT * FROM argo_floats WHERE status = 'active' AND region = 'indian_ocean' LIMIT 50;
        
        Question: "Compare temperature and salinity at different depths"
        SQL: SELECT depth, AVG(temperature) as avg_temp, AVG(salinity) as avg_salinity FROM argo_profiles WHERE temperature IS NOT NULL AND salinity IS NOT NULL GROUP BY CAST(depth/100 AS INTEGER)*100 ORDER BY depth LIMIT 20;
        
        Question: "What are the nearest floats to coordinates 15°N, 70°E?"
        SQL: SELECT float_id, latitude, longitude, status, ABS(latitude - 15) + ABS(longitude - 70) as distance FROM argo_floats ORDER BY distance LIMIT 10;
        
        Now convert this question to SQL:
        Question: {question}
        SQL:"""
        
        return PromptTemplate(
            input_variables=["schema_info", "question"],
            template=template
        )
    
    # ============== ENHANCED POSTGRESQL RAG INTEGRATION ==============
    
    async def ingest_postgresql_data_to_rag(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Ingest real PostgreSQL ARGO data into enhanced RAG system"""
        
        if not self.use_enhanced_features:
            logger.warning("Enhanced features disabled, skipping PostgreSQL RAG ingestion")
            return {}
        
        if not self.db_pool:
            if not await self.initialize_postgresql():
                raise RuntimeError("PostgreSQL connection failed")
        
        logger.info("Starting PostgreSQL ARGO data ingestion for enhanced RAG system...")
        
        ingestion_stats = {
            'floats_processed': 0,
            'profiles_processed': 0,
            'measurements_processed': 0,
            'embeddings_created': 0,
            'metadata_documents': 0,
            'context_documents': 0,
            'regional_documents': 0,
            'temporal_documents': 0,
            'start_time': datetime.now()
        }
        
        async with self.db_pool.acquire() as conn:
            
            # 1. Process Float Metadata Documents
            logger.info("Processing float metadata for RAG...")
            await self._process_postgresql_float_metadata(conn, ingestion_stats)
            
            # 2. Process Profile Summary Documents
            logger.info("Processing profile summaries for RAG...")
            await self._process_postgresql_profile_summaries(conn, ingestion_stats)
            
            # 3. Process Depth-Stratified Measurement Contexts
            logger.info("Processing measurement contexts for RAG...")
            await self._process_postgresql_measurement_contexts(conn, ingestion_stats)
            
            # 4. Process Regional Oceanographic Summaries
            logger.info("Processing regional summaries for RAG...")
            await self._process_postgresql_regional_summaries(conn, ingestion_stats)
            
            # 5. Process Temporal Pattern Documents
            logger.info("Processing temporal patterns for RAG...")
            await self._process_postgresql_temporal_patterns(conn, ingestion_stats)
        
        # Save enhanced cache
        self._save_document_cache()
        
        ingestion_stats['end_time'] = datetime.now()
        ingestion_stats['total_duration'] = (ingestion_stats['end_time'] - ingestion_stats['start_time']).total_seconds()
        
        logger.info(f"Enhanced PostgreSQL RAG ingestion completed: {ingestion_stats}")
        return ingestion_stats
    
    async def _process_postgresql_float_metadata(self, conn, stats: Dict[str, Any]):
        """Process float metadata from PostgreSQL for RAG"""
        
        floats = await conn.fetch("""
            SELECT 
                f.*,
                COUNT(p.profile_id) as actual_profiles,
                MIN(p.profile_datetime) as first_profile,
                MAX(p.profile_datetime) as last_profile,
                AVG(ps.avg_temperature) as overall_avg_temp,
                AVG(ps.avg_salinity) as overall_avg_salinity,
                AVG(ps.max_depth) as avg_max_depth
            FROM floats f
            LEFT JOIN profiles p ON f.float_id = p.float_id
            LEFT JOIN profile_summaries ps ON p.profile_id = ps.profile_id
            GROUP BY f.float_id, f.model, f.platform_number, f.deploy_date, 
                     f.last_lat, f.last_lon, f.region, f.status, f.total_profiles,
                     f.last_profile_date, f.created_at, f.updated_at
        """)
        
        for float_data in floats:
            float_id = float_data['float_id']
            
            # Create comprehensive float summary for RAG
            content = f"""
ARGO Float {float_id} - Comprehensive Operational Summary

Platform Details:
- Float ID: {float_id}
- Model: {float_data['model']}
- Platform Number: {float_data['platform_number']}
- Deployment Date: {float_data['deploy_date']}
- Current Status: {float_data['status']}
- Operating Region: {float_data['region']}

Geographic Information:
- Last Known Position: {float_data['last_lat']:.4f}°N, {float_data['last_lon']:.4f}°E
- Primary Region: {float_data['region']}

Data Collection Performance:
- Planned Profiles: {float_data['total_profiles']}
- Actual Profiles Collected: {float_data['actual_profiles']}
- Collection Success Rate: {(float_data['actual_profiles']/float_data['total_profiles']*100):.1f}%
- Operational Period: {float_data['first_profile']} to {float_data['last_profile']}
- Mission Duration: {(float_data['last_profile'] - float_data['first_profile']).days if float_data['first_profile'] and float_data['last_profile'] else 0} days

Oceanographic Data Summary:
- Average Temperature: {float_data['overall_avg_temp']:.2f}°C
- Average Salinity: {float_data['overall_avg_salinity']:.2f} PSU
- Average Maximum Depth: {float_data['avg_max_depth']:.1f}m

Scientific Contribution:
This ARGO float has been instrumental in providing high-quality oceanographic data from the {float_data['region']}. 
The data contributes to understanding regional water mass properties, circulation patterns, and climate variability. 
The float's {float_data['actual_profiles']} profiles represent a significant dataset for oceanographic research and 
operational oceanography applications including weather prediction and climate monitoring.
            """.strip()
            
            # Create enhanced document for RAG
            doc_id = f"float_metadata_{float_id}"
            embedding = self._generate_enhanced_embedding(content)
            
            document = {
                'id': doc_id,
                'content': content,
                'document_type': 'float_metadata',
                'metadata': {
                    'float_id': float_id,
                    'region': float_data['region'],
                    'model': float_data['model'],
                    'status': float_data['status'],
                    'actual_profiles': float_data['actual_profiles'],
                    'success_rate': float(float_data['actual_profiles']/float_data['total_profiles']*100) if float_data['total_profiles'] > 0 else 0.0,
                    'operational_days': (float_data['last_profile'] - float_data['first_profile']).days if float_data['first_profile'] and float_data['last_profile'] else 0,
                    'avg_temperature': float(float_data['overall_avg_temp']) if float_data['overall_avg_temp'] else None,
                    'avg_salinity': float(float_data['overall_avg_salinity']) if float_data['overall_avg_salinity'] else None,
                    'avg_depth': float(float_data['avg_max_depth']) if float_data['avg_max_depth'] else None
                },
                'embedding': embedding,
                'created_at': datetime.now()
            }
            
            await self._add_enhanced_document(document)
            stats['metadata_documents'] += 1
    
    async def _process_postgresql_profile_summaries(self, conn, stats: Dict[str, Any]):
        """Process profile summaries from PostgreSQL for RAG"""
        
        profiles = await conn.fetch("""
            SELECT 
                p.*,
                ps.*,
                f.region,
                f.model
            FROM profiles p
            JOIN profile_summaries ps ON p.profile_id = ps.profile_id
            JOIN floats f ON p.float_id = f.float_id
            ORDER BY p.profile_datetime
        """)
        
        for profile_data in profiles:
            profile_id = profile_data['profile_id']
            
            # Create detailed profile document for RAG
            content = f"""
ARGO Profile {profile_id} - Detailed Oceanographic Analysis

Profile Identification:
- Profile ID: {profile_id}
- Float: {profile_data['float_id']} ({profile_data['model']})
- Cycle Number: {profile_data['cycle_number']}
- Measurement Date: {profile_data['profile_datetime']}
- Quality Control Status: {profile_data['qc_status']}

Geographic Context:
- Location: {profile_data['lat']:.4f}°N, {profile_data['lon']:.4f}°E
- Ocean Region: {profile_data['region']}

Measurement Summary:
- Total Measurements: {profile_data['measurement_count']}
- Depth Coverage: {profile_data['min_depth']:.1f}m to {profile_data['max_depth']:.1f}m
- Depth Range: {profile_data['max_depth'] - profile_data['min_depth']:.1f}m
- Average Temperature: {profile_data['avg_temperature']:.2f}°C
- Average Salinity: {profile_data['avg_salinity']:.2f} PSU

Oceanographic Significance:
This profile represents a complete vertical sampling of the water column in the {profile_data['region']}. 
The {profile_data['measurement_count']} measurements capture the thermal and haline structure from surface 
waters down to {profile_data['max_depth']:.0f}m depth. The observed temperature range and salinity values 
are characteristic of the regional oceanography, providing insights into water mass properties, 
stratification, and mixing processes.

Data Quality:
Quality control status: {profile_data['qc_status']}. This profile contributes to our understanding of 
oceanic variability and climate patterns in the {profile_data['region']} region.
            """.strip()
            
            # Create enhanced document for RAG
            doc_id = f"profile_summary_{profile_id}"
            embedding = self._generate_enhanced_embedding(content)
            
            document = {
                'id': doc_id,
                'content': content,
                'document_type': 'profile_summary',
                'metadata': {
                    'profile_id': profile_id,
                    'float_id': profile_data['float_id'],
                    'datetime': profile_data['profile_datetime'].isoformat(),
                    'latitude': float(profile_data['lat']),
                    'longitude': float(profile_data['lon']),
                    'region': profile_data['region'],
                    'cycle_number': profile_data['cycle_number'],
                    'measurement_count': profile_data['measurement_count'],
                    'min_depth': float(profile_data['min_depth']),
                    'max_depth': float(profile_data['max_depth']),
                    'temperature_avg': float(profile_data['avg_temperature']),
                    'salinity_avg': float(profile_data['avg_salinity']),
                    'qc_status': profile_data['qc_status']
                },
                'embedding': embedding,
                'created_at': datetime.now()
            }
            
            await self._add_enhanced_document(document)
            stats['metadata_documents'] += 1
    
    async def _process_postgresql_measurement_contexts(self, conn, stats: Dict[str, Any]):
        """Process measurement contexts from PostgreSQL for RAG"""
        
        # Define oceanographic depth layers
        depth_layers = [
            (0, 50, "Surface Mixed Layer"),
            (50, 200, "Subsurface Layer"), 
            (200, 1000, "Intermediate Waters"),
            (1000, 2000, "Deep Waters")
        ]
        
        for min_depth, max_depth, layer_name in depth_layers:
            measurements = await conn.fetch("""
                SELECT 
                    p.float_id,
                    p.profile_id,
                    p.profile_datetime,
                    p.lat,
                    p.lon,
                    f.region,
                    m.depth_m,
                    m.temperature_c,
                    m.salinity_psu,
                    m.pressure_dbar
                FROM measurements m
                JOIN profiles p ON m.profile_id = p.profile_id
                JOIN floats f ON p.float_id = f.float_id
                WHERE m.depth_m >= $1 AND m.depth_m <= $2
                AND m.temperature_c IS NOT NULL 
                AND m.salinity_psu IS NOT NULL
                ORDER BY p.profile_datetime
                LIMIT 1000
            """, min_depth, max_depth)
            
            if measurements:
                # Calculate comprehensive statistics
                temps = [m['temperature_c'] for m in measurements if m['temperature_c']]
                salinities = [m['salinity_psu'] for m in measurements if m['salinity_psu']]
                depths = [m['depth_m'] for m in measurements if m['depth_m']]
                
                content = f"""
{layer_name} Oceanographic Characteristics - Depth Range {min_depth}-{max_depth}m

Layer Definition:
- Depth Range: {min_depth} to {max_depth} meters
- Oceanographic Zone: {layer_name}
- Sample Size: {len(measurements)} measurements

Statistical Summary:
- Temperature Statistics:
  * Range: {min(temps):.2f}°C to {max(temps):.2f}°C
  * Mean: {np.mean(temps):.2f}°C
  * Standard Deviation: ±{np.std(temps):.2f}°C
  * Median: {np.median(temps):.2f}°C

- Salinity Statistics:
  * Range: {min(salinities):.2f} to {max(salinities):.2f} PSU
  * Mean: {np.mean(salinities):.2f} PSU
  * Standard Deviation: ±{np.std(salinities):.2f} PSU
  * Median: {np.median(salinities):.2f} PSU

- Depth Distribution:
  * Mean Depth: {np.mean(depths):.1f}m
  * Depth Standard Deviation: ±{np.std(depths):.1f}m

Temporal Coverage:
- Measurement Period: {min(m['profile_datetime'] for m in measurements)} to {max(m['profile_datetime'] for m in measurements)}
- Data Span: {(max(m['profile_datetime'] for m in measurements) - min(m['profile_datetime'] for m in measurements)).days} days

Oceanographic Context:
The {layer_name.lower()} ({min_depth}-{max_depth}m) represents a critical component of the ocean's vertical structure.
This depth range is characterized by distinct hydrographic properties and plays an important role in 
ocean circulation, heat transport, and biogeochemical processes. The observed temperature and salinity 
characteristics reflect the influence of surface forcing, water mass mixing, and regional circulation patterns.

Regional Significance:
In the Indian Ocean region, this depth layer shows typical characteristics influenced by monsoon dynamics, 
water mass formation, and interaction with adjacent ocean basins. The variability observed in the data 
reflects seasonal cycles, interannual variability, and longer-term climate trends.
                """.strip()
                
                # Create enhanced document for RAG
                doc_id = f"depth_layer_{min_depth}_{max_depth}"
                embedding = self._generate_enhanced_embedding(content)
                
                document = {
                    'id': doc_id,
                    'content': content,
                    'document_type': 'depth_layer_context',
                    'metadata': {
                        'layer_name': layer_name,
                        'min_depth': min_depth,
                        'max_depth': max_depth,
                        'measurement_count': len(measurements),
                        'temp_min': float(min(temps)),
                        'temp_max': float(max(temps)),
                        'temp_mean': float(np.mean(temps)),
                        'temp_std': float(np.std(temps)),
                        'temp_median': float(np.median(temps)),
                        'sal_min': float(min(salinities)),
                        'sal_max': float(max(salinities)),
                        'sal_mean': float(np.mean(salinities)),
                        'sal_std': float(np.std(salinities)),
                        'sal_median': float(np.median(salinities)),
                        'depth_mean': float(np.mean(depths)),
                        'depth_std': float(np.std(depths))
                    },
                    'embedding': embedding,
                    'created_at': datetime.now()
                }
                
                await self._add_enhanced_document(document)
                stats['context_documents'] += 1
    
    async def _process_postgresql_regional_summaries(self, conn, stats: Dict[str, Any]):
        """Process regional oceanographic summaries from PostgreSQL"""
        
        # Get regional statistics
        regional_data = await conn.fetchrow("""
            SELECT 
                f.region,
                COUNT(DISTINCT f.float_id) as float_count,
                COUNT(DISTINCT p.profile_id) as profile_count,
                COUNT(m.meas_id) as measurement_count,
                MIN(p.profile_datetime) as earliest_data,
                MAX(p.profile_datetime) as latest_data,
                AVG(p.lat) as avg_latitude,
                AVG(p.lon) as avg_longitude,
                MIN(p.lat) as min_lat, MAX(p.lat) as max_lat,
                MIN(p.lon) as min_lon, MAX(p.lon) as max_lon,
                AVG(m.temperature_c) as avg_temperature,
                AVG(m.salinity_psu) as avg_salinity,
                AVG(m.depth_m) as avg_depth,
                MAX(m.depth_m) as max_depth
            FROM floats f
            JOIN profiles p ON f.float_id = p.float_id
            JOIN measurements m ON p.profile_id = m.profile_id
            WHERE m.temperature_c IS NOT NULL
            AND m.salinity_psu IS NOT NULL
            GROUP BY f.region
        """)
        
        if regional_data and regional_data['float_count']:
            region = regional_data['region']
            
            content = f"""
{region} Regional Oceanographic Summary:

Geographic Coverage:
- Region: {region}
- Latitude Range: {regional_data['min_lat']:.2f}°N to {regional_data['max_lat']:.2f}°N
- Longitude Range: {regional_data['min_lon']:.2f}°E to {regional_data['max_lon']:.2f}°E
- Central Location: {regional_data['avg_latitude']:.2f}°N, {regional_data['avg_longitude']:.2f}°E

Data Coverage:
- ARGO Floats Deployed: {regional_data['float_count']}
- Total Profiles: {regional_data['profile_count']}
- Total Measurements: {regional_data['measurement_count']}
- Temporal Coverage: {regional_data['earliest_data']} to {regional_data['latest_data']}

Oceanographic Properties:
- Average Temperature: {regional_data['avg_temperature']:.2f}°C
- Average Salinity: {regional_data['avg_salinity']:.2f} PSU
- Average Measurement Depth: {regional_data['avg_depth']:.1f}m
- Maximum Depth Sampled: {regional_data['max_depth']:.1f}m

Regional Characteristics:
The {region} represents a significant oceanic region with unique hydrographic
characteristics. The ARGO float data reveals important information about water mass
properties, circulation patterns, and temporal variability. This region's oceanographic
conditions are influenced by large-scale climate patterns, local forcing, and
interaction with neighboring water masses.
            """.strip()
            
            doc_id = f"region_{region.lower().replace(' ', '_')}"
            embedding = self._generate_enhanced_embedding(content)
            
            document = {
                'id': doc_id,
                'content': content,
                'document_type': 'regional_summary',
                'metadata': {
                    'region': region,
                    'float_count': regional_data['float_count'],
                    'profile_count': regional_data['profile_count'],
                    'measurement_count': regional_data['measurement_count'],
                    'temporal_span_days': (regional_data['latest_data'] - regional_data['earliest_data']).days,
                    'lat_min': float(regional_data['min_lat']),
                    'lat_max': float(regional_data['max_lat']),
                    'lon_min': float(regional_data['min_lon']),
                    'lon_max': float(regional_data['max_lon']),
                    'avg_temperature': float(regional_data['avg_temperature']),
                    'avg_salinity': float(regional_data['avg_salinity']),
                    'avg_depth': float(regional_data['avg_depth'])
                },
                'embedding': embedding,
                'created_at': datetime.now()
            }
            
            await self._add_enhanced_document(document)
            stats['regional_documents'] += 1
    
    async def _process_postgresql_temporal_patterns(self, conn, stats: Dict[str, Any]):
        """Process temporal oceanographic patterns from PostgreSQL"""
        
        # Process seasonal patterns
        seasonal_data = await conn.fetch("""
            SELECT 
                EXTRACT(MONTH FROM p.profile_datetime) as month,
                COUNT(*) as profile_count,
                AVG(m.temperature_c) as avg_temp,
                AVG(m.salinity_psu) as avg_salinity,
                AVG(m.depth_m) as avg_depth
            FROM profiles p
            JOIN measurements m ON p.profile_id = m.profile_id
            WHERE m.temperature_c IS NOT NULL
            AND m.salinity_psu IS NOT NULL
            GROUP BY EXTRACT(MONTH FROM p.profile_datetime)
            ORDER BY month
        """)
        
        if seasonal_data:
            seasons = {
                'Winter': [12, 1, 2],
                'Spring': [3, 4, 5], 
                'Summer': [6, 7, 8],
                'Autumn': [9, 10, 11]
            }
            
            for season_name, months in seasons.items():
                season_stats = [row for row in seasonal_data if row['month'] in months]
                
                if season_stats:
                    total_profiles = sum(row['profile_count'] for row in season_stats)
                    avg_temp = np.mean([row['avg_temp'] for row in season_stats])
                    avg_salinity = np.mean([row['avg_salinity'] for row in season_stats])
                    
                    content = f"""
{season_name} Seasonal Oceanographic Patterns:

Seasonal Overview:
- Season: {season_name} (Months: {', '.join(map(str, months))})
- Total Profiles: {total_profiles}
- Average Temperature: {avg_temp:.2f}°C
- Average Salinity: {avg_salinity:.2f} PSU

Seasonal Characteristics:
During {season_name.lower()}, the ocean exhibits distinctive patterns in temperature
and salinity distribution. These seasonal variations are driven by changes in
surface forcing (heating/cooling, freshwater fluxes), wind patterns, and
large-scale circulation changes.

Oceanographic Significance:
The observed {season_name.lower()} patterns reflect the ocean's response to
seasonal atmospheric forcing and provide insights into the annual cycle of
ocean-atmosphere interaction in this region.
                    """.strip()
                    
                    doc_id = f"seasonal_{season_name.lower()}"
                    embedding = self._generate_enhanced_embedding(content)
                    
                    document = {
                        'id': doc_id,
                        'content': content,
                        'document_type': 'seasonal_pattern',
                        'metadata': {
                            'season': season_name,
                            'months_str': ','.join(map(str, months)),
                            'profile_count': total_profiles,
                            'avg_temperature': float(avg_temp),
                            'avg_salinity': float(avg_salinity)
                        },
                        'embedding': embedding,
                        'created_at': datetime.now()
                    }
                    
                    await self._add_enhanced_document(document)
                    stats['temporal_documents'] += 1
    
    async def _add_enhanced_document(self, document: Dict[str, Any]):
        """Add document to enhanced RAG system"""
        
        # Add to cache
        self.document_cache[document['id']] = document
        
        # Add to enhanced vector database
        if self.enhanced_collection:
            try:
                self.enhanced_collection.add(
                    embeddings=[document['embedding']],
                    documents=[document['content']],
                    metadatas=[document['metadata']],
                    ids=[document['id']]
                )
                logger.debug(f"Added document {document['id']} to enhanced vector database")
            except Exception as e:
                logger.warning(f"Failed to add document to enhanced ChromaDB: {str(e)}")
    
    # ============== END ENHANCED POSTGRESQL RAG INTEGRATION ==============
    
    def add_data_to_vector_store(self, profiles: List[Dict], floats: List[Dict]):
        """Add ARGO data to vector store for retrieval"""
        try:
            # Create documents for vector store
            documents = []
            
            # Add profile documents
            for profile in profiles:
                content = f"""
                Profile ID: {profile.get('profile_id', 'N/A')}
                Float ID: {profile.get('float_id', 'N/A')}
                Location: {profile.get('latitude', 0):.2f}°N, {profile.get('longitude', 0):.2f}°E
                Date: {profile.get('timestamp', 'N/A')}
                Depth: {profile.get('depth', 0):.1f}m
                Temperature: {profile.get('temperature', 'N/A')}°C
                Salinity: {profile.get('salinity', 'N/A')} PSU
                Region: {profile.get('region', 'N/A')}
                """
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "type": "profile",
                        "profile_id": profile.get('profile_id'),
                        "float_id": profile.get('float_id'),
                        "region": profile.get('region'),
                        "latitude": profile.get('latitude'),
                        "longitude": profile.get('longitude'),
                        "timestamp": profile.get('timestamp')
                    }
                )
                documents.append(doc)
            
            # Add float documents
            for float_data in floats:
                content = f"""
                Float ID: {float_data.get('float_id', 'N/A')}
                WMO ID: {float_data.get('wmo_id', 'N/A')}
                Status: {float_data.get('status', 'N/A')}
                Region: {float_data.get('region', 'N/A')}
                Location: {float_data.get('latitude', 0):.2f}°N, {float_data.get('longitude', 0):.2f}°E
                Deployment Date: {float_data.get('deployment_date', 'N/A')}
                Total Profiles: {float_data.get('total_profiles', 0)}
                """
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "type": "float",
                        "float_id": float_data.get('float_id'),
                        "status": float_data.get('status'),
                        "region": float_data.get('region'),
                        "latitude": float_data.get('latitude'),
                        "longitude": float_data.get('longitude')
                    }
                )
                documents.append(doc)
            
            # Initialize Chroma vector store
            if documents:
                self.chroma_db = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=config.CHROMA_PERSIST_DIRECTORY
                )
                print(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            print(f"Error adding data to vector store: {e}")
    
    def generate_sql_from_nl(self, question: str) -> str:
        """Convert natural language question to SQL query"""
        try:
            if self.llm:
                # Use LLM for SQL generation
                prompt = self.nl_to_sql_prompt.format(
                    schema_info=self.schema_info,
                    question=question
                )
                response = self.llm.predict(prompt)
                sql = response.strip()
                
                # Clean up the response
                if sql.startswith("SQL:"):
                    sql = sql[4:].strip()
                
                return sql
            else:
                # Use rule-based approach as fallback
                return self._rule_based_sql_generation(question)
                
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return self._rule_based_sql_generation(question)
    
    def _rule_based_sql_generation(self, question: str) -> str:
        """Fallback rule-based SQL generation"""
        question_lower = question.lower()
        
        # Common patterns
        if "salinity" in question_lower and "profile" in question_lower:
            if "equator" in question_lower:
                return "SELECT profile_id, latitude, longitude, depth, salinity, timestamp FROM argo_profiles WHERE latitude BETWEEN -5 AND 5 AND salinity IS NOT NULL ORDER BY timestamp LIMIT 100;"
            else:
                return "SELECT profile_id, latitude, longitude, depth, salinity, timestamp FROM argo_profiles WHERE salinity IS NOT NULL ORDER BY timestamp LIMIT 100;"
        
        elif "temperature" in question_lower:
            if "indian ocean" in question_lower:
                return "SELECT profile_id, latitude, longitude, depth, temperature, timestamp FROM argo_profiles WHERE region = 'indian_ocean' AND temperature IS NOT NULL ORDER BY timestamp LIMIT 100;"
            else:
                return "SELECT profile_id, latitude, longitude, depth, temperature, timestamp FROM argo_profiles WHERE temperature IS NOT NULL ORDER BY timestamp LIMIT 100;"
        
        elif "float" in question_lower and ("active" in question_lower or "status" in question_lower):
            return "SELECT * FROM argo_floats WHERE status = 'active' LIMIT 50;"
        
        elif "bgc" in question_lower or "oxygen" in question_lower or "chlorophyll" in question_lower:
            return "SELECT profile_id, latitude, longitude, depth, oxygen, chlorophyll, nitrate, ph, timestamp FROM argo_profiles WHERE oxygen IS NOT NULL OR chlorophyll IS NOT NULL OR nitrate IS NOT NULL ORDER BY timestamp LIMIT 100;"
        
        elif "compare" in question_lower:
            if "temperature" in question_lower and "salinity" in question_lower:
                return "SELECT depth, AVG(temperature) as avg_temp, AVG(salinity) as avg_salinity FROM argo_profiles WHERE temperature IS NOT NULL AND salinity IS NOT NULL GROUP BY CAST(depth/100 AS INTEGER)*100 ORDER BY depth LIMIT 20;"
        
        elif "nearest" in question_lower or "near" in question_lower:
            # Extract coordinates if possible
            coord_match = re.search(r'(\d+)°?[NS].*?(\d+)°?[EW]', question)
            if coord_match:
                lat, lon = coord_match.groups()
                return f"SELECT float_id, latitude, longitude, status, ABS(latitude - {lat}) + ABS(longitude - {lon}) as distance FROM argo_floats ORDER BY distance LIMIT 10;"
        
        elif "recent" in question_lower or "last" in question_lower:
            return "SELECT * FROM argo_profiles WHERE timestamp > datetime('now', '-30 days') ORDER BY timestamp DESC LIMIT 100;"
        
        # Default query
        return "SELECT * FROM argo_profiles ORDER BY timestamp DESC LIMIT 50;"
    
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a natural language query and return results"""
        try:
            question = request.query
            
            # Determine query type
            query_type = self._classify_query(question)
            
            # Generate SQL
            sql_query = self.generate_sql_from_nl(question)
            
            # Execute query (for now, return mock results)
            results = self._execute_mock_query(sql_query, question)
            
            # Generate natural language response
            natural_response = self._generate_natural_response(question, results, query_type)
            
            # Determine visualization type
            viz_type = self._determine_visualization_type(question, results)
            
            return QueryResponse(
                query=question,
                sql_query=sql_query,
                results=results,
                visualization_type=viz_type,
                metadata={
                    "query_type": query_type,
                    "natural_response": natural_response,
                    "result_count": len(results),
                    "execution_time": "0.1s",
                    "suggestions": self._generate_suggestions(question)
                }
            )
            
        except Exception as e:
            return QueryResponse(
                query=request.query,
                sql_query="-- Error generating query",
                results=[],
                metadata={
                    "error": str(e),
                    "natural_response": f"I encountered an error processing your query: {str(e)}"
                }
            )
    
    def _classify_query(self, question: str) -> str:
        """Classify the type of query"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["plot", "graph", "chart", "visualize", "show"]):
            return "visualization"
        elif any(word in question_lower for word in ["compare", "correlation", "relationship"]):
            return "analysis"
        elif any(word in question_lower for word in ["float", "buoy", "instrument"]):
            return "instrument"
        elif any(word in question_lower for word in ["profile", "depth", "vertical"]):
            return "profile"
        elif any(word in question_lower for word in ["map", "location", "coordinates", "nearest"]):
            return "spatial"
        elif any(word in question_lower for word in ["recent", "time", "temporal", "trend"]):
            return "temporal"
        else:
            return "general"
    
    def _execute_mock_query(self, sql_query: str, question: str) -> List[Dict]:
        """Execute mock query for demonstration (replace with actual DB execution)"""
        # This would normally execute against a real database
        # For now, return mock data based on the query type
        
        question_lower = question.lower()
        
        if "salinity" in question_lower:
            return [
                {
                    "profile_id": f"ARGO100001_P{i:03d}",
                    "float_id": "ARGO100001",
                    "latitude": 0.5 + i * 0.1,
                    "longitude": 65.0 + i * 0.2,
                    "depth": i * 10,
                    "salinity": 35.0 + i * 0.01,
                    "temperature": 28.0 - i * 0.1,
                    "timestamp": (datetime.now() - timedelta(days=i)).isoformat()
                }
                for i in range(20)
            ]
        
        elif "temperature" in question_lower:
            return [
                {
                    "profile_id": f"ARGO100002_P{i:03d}",
                    "float_id": "ARGO100002",
                    "latitude": 15.0 + i * 0.05,
                    "longitude": 70.0 + i * 0.1,
                    "depth": i * 25,
                    "temperature": 28.0 - i * 0.15,
                    "salinity": 35.2 + i * 0.005,
                    "timestamp": (datetime.now() - timedelta(days=i)).isoformat()
                }
                for i in range(15)
            ]
        
        elif "float" in question_lower:
            return [
                {
                    "float_id": f"ARGO{100000 + i}",
                    "wmo_id": f"{1900000 + i}",
                    "status": "active" if i % 3 == 0 else "inactive",
                    "latitude": 15.0 + i * 2,
                    "longitude": 70.0 + i * 1.5,
                    "region": "indian_ocean",
                    "total_profiles": 150 + i * 10,
                    "deployment_date": (datetime.now() - timedelta(days=365 + i * 30)).isoformat()
                }
                for i in range(10)
            ]
        
        else:
            # General query
            return [
                {
                    "profile_id": f"ARGO100000_P{i:03d}",
                    "float_id": "ARGO100000",
                    "latitude": 10.0 + i * 0.1,
                    "longitude": 75.0 + i * 0.1,
                    "depth": i * 50,
                    "temperature": 27.0 - i * 0.2,
                    "salinity": 35.1 + i * 0.01,
                    "pressure": i * 5,
                    "timestamp": (datetime.now() - timedelta(hours=i * 6)).isoformat(),
                    "region": "indian_ocean",
                    "quality_flag": 1
                }
                for i in range(25)
            ]
    
    def _generate_natural_response(self, question: str, results: List[Dict], query_type: str) -> str:
        """Generate natural language response"""
        result_count = len(results)
        
        if result_count == 0:
            return f"I couldn't find any data matching your query about {question}. Try rephrasing or broadening your search criteria."
        
        base_response = f"I found {result_count} results for your query about {question}."
        
        if query_type == "profile":
            return f"{base_response} The data includes oceanographic profiles with depth measurements ranging from the surface to deep waters."
        elif query_type == "spatial":
            return f"{base_response} The results show data from various geographic locations across the specified region."
        elif query_type == "temporal":
            return f"{base_response} The temporal data spans recent measurements with detailed time series information."
        elif query_type == "instrument":
            return f"{base_response} The float information includes deployment status, location, and operational history."
        elif query_type == "analysis":
            return f"{base_response} The analysis shows relationships between different oceanographic parameters."
        elif query_type == "visualization":
            return f"{base_response} The data is ready for visualization showing the requested parameters."
        else:
            return f"{base_response} The dataset includes comprehensive oceanographic measurements from ARGO floats."
    
    def _determine_visualization_type(self, question: str, results: List[Dict]) -> Optional[str]:
        """Determine appropriate visualization type"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["map", "location", "spatial", "geographic"]):
            return "map"
        elif any(word in question_lower for word in ["profile", "depth", "vertical"]):
            return "profile"
        elif any(word in question_lower for word in ["time", "temporal", "trend", "series"]):
            return "time_series"
        elif any(word in question_lower for word in ["compare", "correlation", "vs", "against"]):
            return "comparison"
        elif "plot" in question_lower:
            # Try to extract what to plot
            if " vs " in question_lower:
                return "scatter"
            elif any(param in question_lower for param in ["temperature", "salinity", "oxygen"]):
                return "line"
        
        # Default based on data type
        if results and len(results) > 0:
            if "latitude" in results[0] and "longitude" in results[0]:
                return "map"
            elif "depth" in results[0]:
                return "profile"
            else:
                return "table"
        
        return None
    
    def _generate_suggestions(self, question: str) -> List[str]:
        """Generate query suggestions"""
        suggestions = [
            "Show me temperature profiles in the Indian Ocean",
            "Find active ARGO floats near the equator",
            "Compare salinity and temperature at different depths",
            "Plot temperature vs salinity for recent data",
            "Show BGC parameters from the Arabian Sea"
        ]
        return suggestions[:3]  # Return top 3 suggestions
    
    def search_similar_profiles(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar profiles using vector similarity"""
        try:
            if self.chroma_db:
                docs = self.chroma_db.similarity_search(query, k=k)
                return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
            else:
                return []
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_retrieval_chain(self):
        """Get retrieval chain for question answering"""
    def get_retrieval_chain(self):
        """Get retrieval chain for question answering"""
        if self.chroma_db and self.llm:
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.chroma_db.as_retriever(search_kwargs={"k": 5}),
                memory=self.memory
            )
        return None

    def _setup_retrieval_chain(self):
        """Setup the retrieval-augmented generation chain"""
        try:
            # Create a simple vector store for the LLM
            documents = self._create_documents_from_vector_store()
            if documents:
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                
                self.retrieval_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                    return_source_documents=True
                )
        except Exception as e:
            print(f"Warning: Could not setup retrieval chain: {e}")
            self.retrieval_chain = None

    def _create_documents_from_vector_store(self) -> List[Document]:
        """Create LangChain documents from vector store data"""
        documents = []
        
        try:
            # Get sample data from vector store
            profile_results = self.vector_store.search_profiles("ARGO profile data", n_results=50)
            float_results = self.vector_store.search_floats("ARGO float data", n_results=20)
            
            for result in profile_results:
                doc = Document(
                    page_content=result["document"],
                    metadata=result["metadata"]
                )
                documents.append(doc)
            
            for result in float_results:
                doc = Document(
                    page_content=result["document"],
                    metadata=result["metadata"]
                )
                documents.append(doc)
                
        except Exception as e:
            print(f"Warning: Could not create documents from vector store: {e}")
        
        return documents

    def process_natural_language_query(self, query: str, limit: int = 100) -> QueryResponse:
        """Process a natural language query and return structured response"""
        # First, use the query processor to extract structured information
        processed_query = self.query_processor.process_query(query, limit)
        
        # Generate a natural language response using LLM if available
        natural_response = self._generate_llm_response(query, processed_query)
        
        # For demo purposes, return mock data (in real implementation, this would query the database)
        mock_results = self._generate_mock_results(processed_query, limit)
        
        return QueryResponse(
            query=query,
            sql_query=processed_query["sql_query"],
            results=mock_results,
            visualization_type=processed_query["visualization_type"],
            metadata={
                "natural_response": natural_response,
                "extracted_info": processed_query["metadata"],
                "processing_method": "rag_pipeline"
            }
        )

    def _generate_llm_response(self, query: str, processed_query: Dict) -> str:
        """Generate a natural language response using LLM"""
        if not self.llm:
            return self._generate_fallback_response(query, processed_query)
        
        try:
            # Create a context from the processed query
            context = f"""
            User Query: {query}
            
            Extracted Information:
            - Location: {processed_query['metadata']['extracted_location']}
            - Parameters: {processed_query['metadata']['extracted_parameters']}
            - Time Range: {processed_query['metadata']['extracted_time']}
            - Query Type: {processed_query['metadata']['query_type']}
            
            Please provide a helpful response about ARGO oceanographic data based on this query.
            """
            
            # Use LangChain invoke-compatible call and normalize output
            try:
                result = self.llm.invoke(context)
            except Exception:
                result = self.llm(context)
            if isinstance(result, str):
                return result.strip()
            # ChatGoogleGenerativeAI may return an object with .content or .text or dict
            text = None
            for key in ("content", "text", "generation_info"):
                if hasattr(result, key):
                    val = getattr(result, key)
                    if isinstance(val, str) and val.strip():
                        text = val.strip()
                        break
            if text:
                return text
            if isinstance(result, dict):
                for key in ("content", "text", "result"):
                    if isinstance(result.get(key), str) and result.get(key).strip():
                        return result.get(key).strip()
            return self._generate_fallback_response(query, processed_query)
            
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(query, processed_query)

    def _generate_fallback_response(self, query: str, processed_query: Dict) -> str:
        """Generate a fallback response without LLM"""
        location = processed_query['metadata']['extracted_location']
        parameters = processed_query['metadata']['extracted_parameters']
        time_info = processed_query['metadata']['extracted_time']
        
        response_parts = ["I'll help you find ARGO oceanographic data."]
        
        if location.get('region'):
            response_parts.append(f"Searching in the {location['region'].replace('_', ' ').title()} region.")
        
        if parameters:
            response_parts.append(f"Looking for {', '.join(parameters)} measurements.")
        
        if time_info.get('time_range_days'):
            response_parts.append(f"Filtering for the last {time_info['time_range_days']} days.")
        elif time_info.get('start_date'):
            response_parts.append(f"Filtering for data from {time_info['start_date'].strftime('%B %Y')}.")
        
        response_parts.append("Here are the results I found:")
        
        return " ".join(response_parts)

    def _generate_mock_results(self, processed_query: Dict, limit: int) -> List[Dict[str, Any]]:
        """Generate mock results for demonstration (replace with actual database query)"""
        import random
        from datetime import datetime, timedelta
        
        mock_results = []
        num_results = min(limit, random.randint(5, 20))
        
        # Get region from processed query
        region = processed_query['metadata']['extracted_location'].get('region', 'indian_ocean')
        
        for i in range(num_results):
            # Generate mock coordinates based on region
            if region == 'indian_ocean':
                lat = random.uniform(-30, 30)
                lon = random.uniform(20, 120)
            else:
                lat = random.uniform(-60, 60)
                lon = random.uniform(-180, 180)
            
            # Generate mock timestamp
            days_ago = random.randint(1, 365)
            timestamp = datetime.now() - timedelta(days=days_ago)
            
            result = {
                "profile_id": f"ARGO{random.randint(100000, 999999)}_{random.randint(1000, 9999)}",
                "float_id": f"ARGO{random.randint(100000, 999999)}",
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "timestamp": timestamp.isoformat(),
                "depth": random.uniform(0, 2000),
                "temperature": round(random.uniform(0, 30), 2),
                "salinity": round(random.uniform(32, 37), 2),
                "pressure": round(random.uniform(0, 200), 1),
                "region": region
            }
            
            # Add BGC parameters if requested
            parameters = processed_query['metadata']['extracted_parameters']
            if 'oxygen' in parameters:
                result['oxygen'] = round(random.uniform(150, 250), 1)
            if 'chlorophyll' in parameters:
                result['chlorophyll'] = round(random.uniform(0.1, 2.0), 3)
            if 'nitrate' in parameters:
                result['nitrate'] = round(random.uniform(0.5, 5.0), 2)
            if 'ph' in parameters:
                result['ph'] = round(random.uniform(7.8, 8.3), 2)
            
            mock_results.append(result)
        
        return mock_results

    def search_similar_profiles(self, query: str, n_results: int = 10) -> List[Dict]:
        """Search for similar profiles using vector similarity"""
        try:
            results = self.vector_store.search_profiles(query, n_results)
            return results
        except Exception as e:
            print(f"Error searching profiles: {e}")
            return []

    def search_similar_floats(self, query: str, n_results: int = 10) -> List[Dict]:
        """Search for similar floats using vector similarity"""
        try:
            results = self.vector_store.search_floats(query, n_results)
            return results
        except Exception as e:
            print(f"Error searching floats: {e}")
            return []

    def get_context_for_query(self, query: str) -> str:
        """Get relevant context for a query using RAG"""
        if not self.retrieval_chain:
            return "RAG system not available. Using basic query processing."
        
        try:
            # Prefer invoke to avoid deprecated __call__
            try:
                result = self.retrieval_chain.invoke({"query": query})
            except Exception:
                result = self.retrieval_chain({"query": query})
            if isinstance(result, dict) and isinstance(result.get("result"), str):
                return result["result"]
            if isinstance(result, str):
                return result
            return str(result)
        except Exception as e:
            print(f"Error getting context: {e}")
            return "Error retrieving context."

    def add_data_to_vector_store(self, profiles: List[Dict], floats: List[Dict]):
        """Add data to the vector store for RAG"""
        try:
            # Convert to proper objects if needed
            from models.data_models import ArgoProfile, ArgoFloat
            
            profile_objects = []
            for profile_data in profiles:
                if isinstance(profile_data, dict):
                    profile_objects.append(ArgoProfile(**profile_data))
                else:
                    profile_objects.append(profile_data)
            
            float_objects = []
            for float_data in floats:
                if isinstance(float_data, dict):
                    float_objects.append(ArgoFloat(**float_data))
                else:
                    float_objects.append(float_data)
            
            # Add to vector store
            self.vector_store.add_profiles(profile_objects)
            self.vector_store.add_floats(float_objects)
            
            # Recreate retrieval chain with new data
            if self.llm and self.embeddings:
                self._setup_retrieval_chain()
                
        except Exception as e:
            print(f"Error adding data to vector store: {e}")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        stats = {
            "vector_store_stats": self.vector_store.get_collection_stats(),
            "llm_available": self.llm is not None,
            "embeddings_available": self.embeddings is not None,
            "retrieval_chain_available": self.retrieval_chain is not None
        }
        return stats
