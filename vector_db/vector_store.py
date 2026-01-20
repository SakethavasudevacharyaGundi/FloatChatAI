import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
import json
import os
from models.data_models import ArgoProfile, ArgoFloat

class VectorStore:
    """Simple vector store for testing compatibility"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the vector store"""
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
    
    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Query the vector store"""
        # Simple keyword-based search for testing
        results = []
        for i, doc in enumerate(self.documents[:n_results]):
            results.append({
                'document': doc,
                'metadata': self.metadata[i] if i < len(self.metadata) else {},
                'distance': 0.5  # Dummy distance
            })
        return results

class ArgoVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create collections
        self.profiles_collection = self.client.get_or_create_collection(
            name="argo_profiles",
            metadata={"description": "ARGO profile metadata and summaries"}
        )
        
        self.floats_collection = self.client.get_or_create_collection(
            name="argo_floats",
            metadata={"description": "ARGO float metadata and summaries"}
        )

    def create_profile_summary(self, profile: ArgoProfile) -> str:
        """Create a text summary of an ARGO profile for vector storage"""
        summary_parts = [
            f"ARGO Profile {profile.profile_id} from float {profile.float_id}",
            f"Location: {profile.latitude:.2f}°N, {profile.longitude:.2f}°E",
            f"Date: {profile.timestamp.strftime('%Y-%m-%d')}",
            f"Region: {profile.region.value.replace('_', ' ').title()}",
            f"Depth range: {min(profile.depth_levels):.0f}m to {max(profile.depth_levels):.0f}m",
            f"Temperature range: {min(profile.temperature):.2f}°C to {max(profile.temperature):.2f}°C",
            f"Salinity range: {min(profile.salinity):.2f} to {max(profile.salinity):.2f}",
        ]
        
        if profile.oxygen:
            summary_parts.append(f"Oxygen range: {min(profile.oxygen):.1f} to {max(profile.oxygen):.1f} μmol/kg")
        if profile.chlorophyll:
            summary_parts.append(f"Chlorophyll range: {min(profile.chlorophyll):.3f} to {max(profile.chlorophyll):.3f} mg/m³")
        if profile.nitrate:
            summary_parts.append(f"Nitrate range: {min(profile.nitrate):.2f} to {max(profile.nitrate):.2f} μmol/kg")
        if profile.ph:
            summary_parts.append(f"pH range: {min(profile.ph):.2f} to {max(profile.ph):.2f}")
        
        return " | ".join(summary_parts)

    def create_float_summary(self, float_obj: ArgoFloat) -> str:
        """Create a text summary of an ARGO float for vector storage"""
        summary_parts = [
            f"ARGO Float {float_obj.float_id} (WMO: {float_obj.wmo_id})",
            f"Status: {float_obj.status.value.title()}",
            f"Region: {float_obj.region.value.replace('_', ' ').title()}",
            f"Deployed: {float_obj.deployment_date.strftime('%Y-%m-%d')}",
            f"Total profiles: {float_obj.total_profiles}",
        ]
        
        if float_obj.last_profile_date:
            summary_parts.append(f"Last profile: {float_obj.last_profile_date.strftime('%Y-%m-%d')}")
        
        if float_obj.metadata:
            if "institution" in float_obj.metadata:
                summary_parts.append(f"Institution: {float_obj.metadata['institution']}")
            if "country" in float_obj.metadata:
                summary_parts.append(f"Country: {float_obj.metadata['country']}")
        
        return " | ".join(summary_parts)

    def add_profiles(self, profiles: List[ArgoProfile]):
        """Add ARGO profiles to the vector store"""
        if not profiles:
            return
        
        ids = []
        documents = []
        metadatas = []
        
        for profile in profiles:
            ids.append(profile.profile_id)
            documents.append(self.create_profile_summary(profile))
            
            metadata = {
                "float_id": profile.float_id,
                "latitude": profile.latitude,
                "longitude": profile.longitude,
                "timestamp": profile.timestamp.isoformat(),
                "region": profile.region.value,
                "depth_min": min(profile.depth_levels),
                "depth_max": max(profile.depth_levels),
                "temp_min": min(profile.temperature),
                "temp_max": max(profile.temperature),
                "sal_min": min(profile.salinity),
                "sal_max": max(profile.salinity),
                "has_oxygen": profile.oxygen is not None,
                "has_chlorophyll": profile.chlorophyll is not None,
                "has_nitrate": profile.nitrate is not None,
                "has_ph": profile.ph is not None,
                "quality_flag": profile.quality_flag
            }
            metadatas.append(metadata)
        
        self.profiles_collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Added {len(profiles)} profiles to vector store")

    def add_floats(self, floats: List[ArgoFloat]):
        """Add ARGO floats to the vector store"""
        if not floats:
            return
        
        ids = []
        documents = []
        metadatas = []
        
        for float_obj in floats:
            ids.append(float_obj.float_id)
            documents.append(self.create_float_summary(float_obj))
            
            metadata = {
                "wmo_id": float_obj.wmo_id,
                "status": float_obj.status.value,
                "region": float_obj.region.value,
                "deployment_date": float_obj.deployment_date.isoformat(),
                "total_profiles": float_obj.total_profiles,
                "last_profile_date": float_obj.last_profile_date.isoformat() if float_obj.last_profile_date else None,
                "institution": float_obj.metadata.get("institution", ""),
                "country": float_obj.metadata.get("country", "")
            }
            metadatas.append(metadata)
        
        self.floats_collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Added {len(floats)} floats to vector store")

    def search_profiles(self, query: str, n_results: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Search for profiles using semantic similarity"""
        where_clause = {}
        if filters:
            for key, value in filters.items():
                if key in ["region", "has_oxygen", "has_chlorophyll", "has_nitrate", "has_ph"]:
                    where_clause[key] = value
        
        results = self.profiles_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        return self._format_search_results(results)

    def search_floats(self, query: str, n_results: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Search for floats using semantic similarity"""
        where_clause = {}
        if filters:
            for key, value in filters.items():
                if key in ["status", "region", "institution", "country"]:
                    where_clause[key] = value
        
        results = self.floats_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        return self._format_search_results(results)

    def _format_search_results(self, results: Dict) -> List[Dict]:
        """Format search results for easier consumption"""
        formatted_results = []
        
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None
                }
                formatted_results.append(result)
        
        return formatted_results

    def get_profile_by_id(self, profile_id: str) -> Optional[Dict]:
        """Get a specific profile by ID"""
        results = self.profiles_collection.get(ids=[profile_id])
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "document": results["documents"][0],
                "metadata": results["metadatas"][0]
            }
        return None

    def get_float_by_id(self, float_id: str) -> Optional[Dict]:
        """Get a specific float by ID"""
        results = self.floats_collection.get(ids=[float_id])
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "document": results["documents"][0],
                "metadata": results["metadatas"][0]
            }
        return None

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about the collections"""
        return {
            "profiles_count": self.profiles_collection.count(),
            "floats_count": self.floats_collection.count()
        }

    def list_profiles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List a sample of profiles from the vector store (approximate)."""
        try:
            count = max(1, min(limit, self.profiles_collection.count()))
            results = self.profiles_collection.query(
                query_texts=["ARGO profile"],
                n_results=count
            )
            return self._format_search_results(results)
        except Exception:
            return []

    def list_floats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List a sample of floats from the vector store (approximate)."""
        try:
            count = max(1, min(limit, self.floats_collection.count()))
            results = self.floats_collection.query(
                query_texts=["ARGO float"],
                n_results=count
            )
            return self._format_search_results(results)
        except Exception:
            return []

    def find_nearest_float(self, target_lat: float, target_lon: float, target_date_iso: Optional[str] = None, sample_size: int = 500) -> Optional[Dict[str, Any]]:
        """Approximate nearest float by scanning profile metadata and picking the closest by haversine distance.
        Returns a dict with float_id, profile_id, distance_km, and source metadata.
        """
        import math

        def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            R = 6371.0
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return R * c

        # Pull a sample of profiles
        profiles = self.search_profiles("ARGO profile data", n_results=sample_size)
        if not profiles:
            return None

        nearest = None
        best_dist = float("inf")
        for item in profiles:
            meta = item.get("metadata", {})
            lat = meta.get("latitude")
            lon = meta.get("longitude")
            if lat is None or lon is None:
                continue
            d = haversine(float(target_lat), float(target_lon), float(lat), float(lon))
            if d < best_dist:
                best_dist = d
                nearest = {
                    "float_id": meta.get("float_id"),
                    "profile_id": item.get("id"),
                    "distance_km": round(d, 2),
                    "metadata": meta,
                    "document": item.get("document")
                }

        return nearest