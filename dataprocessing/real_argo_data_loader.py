"""
Real ARGO Data Loader - Process actual ARGO CSV data with BGC parameters
Handles the 1900121_prof.csv file with real oceanographic measurements
Enhanced with Bio-Geo-Chemical parameter generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import sqlite3
import pickle
from dataclasses import dataclass
import sys
import os

# Add bgc_generator to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bgc_generator import bgc_generator

logger = logging.getLogger(__name__)

@dataclass
class ArgoProfile:
    """Real ARGO profile data structure"""
    profile_id: int
    float_id: str
    latitude: float
    longitude: float
    date: datetime
    measurements: List[Dict[str, float]]
    
    @property
    def temperature(self) -> Optional[float]:
        """Surface temperature from first measurement"""
        return self.measurements[0]["temperature"] if self.measurements else None
    
    @property
    def salinity(self) -> Optional[float]:
        """Surface salinity from first measurement"""
        return self.measurements[0]["salinity"] if self.measurements else None
    
    @property
    def pressure(self) -> Optional[float]:
        """Surface pressure from first measurement"""
        return self.measurements[0]["pressure"] if self.measurements else None
    
    @property
    def max_depth(self) -> Optional[float]:
        """Maximum depth (pressure) in this profile"""
        return max([m["pressure"] for m in self.measurements]) if self.measurements else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to format expected by RAG pipeline"""
        # Extract depth levels, temperatures, salinities, and pressures
        depth_levels = [m["pressure"] for m in self.measurements]  # Using pressure as depth
        temperatures = [m["temperature"] for m in self.measurements]
        salinities = [m["salinity"] for m in self.measurements] 
        pressures = [m["pressure"] for m in self.measurements]
        
        # Determine region based on coordinates (Indian Ocean)
        region = "indian_ocean"  # Float 1900121 is in Indian Ocean
        
        return {
            "profile_id": str(self.profile_id),  # Convert to string
            "float_id": self.float_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.date,  # Use date as timestamp
            "depth_levels": depth_levels,
            "temperature": temperatures,
            "salinity": salinities,
            "pressure": pressures,
            "region": region,
            "quality_flag": 1,  # Assume good quality
            # Additional fields for compatibility
            "surface_temperature": self.temperature,
            "surface_salinity": self.salinity,
            "max_depth": self.max_depth,
            "total_levels": len(self.measurements)
        }

class RealArgoDataLoader:
    """Load and process real ARGO data from CSV file"""
    
    def __init__(self, data_file: str = "data/1900121_prof.csv"):
        self.data_file = Path(data_file)
        self.db_path = Path("data/real_argo.db")
        self.cache_path = Path("data/argo_cache.pkl")
        self._data = None
        self._profiles = None
    
    def load_data(self) -> pd.DataFrame:
        """Load and process the real ARGO CSV data"""
        if self._data is not None:
            return self._data
            
        try:
            logger.info(f"Loading real ARGO data from {self.data_file}")
            
            # Read the CSV file
            df = pd.read_csv(self.data_file)
            logger.info(f"Raw CSV data: {len(df)} rows, columns: {list(df.columns)}")
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Clean and validate the data
            df = self._clean_data(df)
            
            self._data = df
            logger.info(f"Loaded {len(self._data)} measurements from {self._data['profile_index'].nunique()} profiles")
            logger.info(f"Date range: {self._data['date'].min()} to {self._data['date'].max()}")
            logger.info(f"Geographic coverage: Lat {self._data['latitude'].min():.2f} to {self._data['latitude'].max():.2f}")
            logger.info(f"Temperature range: {self._data['temperature'].min():.1f}°C to {self._data['temperature'].max():.1f}°C")
            
            return self._data
            
        except Exception as e:
            logger.error(f"Error loading ARGO data: {e}")
            raise

class RealArgoDataLoader:
    """Load and process real ARGO data from CSV file"""
    
    def __init__(self, data_file: str = "data/1900121_prof.csv"):
        self.data_file = Path(data_file)
        self.db_path = Path("data/real_argo.db")
        self.cache_path = Path("data/argo_cache.pkl")
        self._data = None
        self._profiles = None
        
    def load_data(self) -> pd.DataFrame:
        """Load real ARGO data from CSV"""
        if self._data is not None:
            return self._data
            
        logger.info(f"Loading real ARGO data from {self.data_file}")
        
        try:
            # Load CSV with proper data types
            self._data = pd.read_csv(self.data_file)
            
            # Convert date columns to datetime
            self._data['date'] = pd.to_datetime(self._data['date'])
            self._data['julian_time'] = pd.to_datetime(self._data['julian_time'])
            
            # Clean and validate data
            self._data = self._clean_data(self._data)
            
            logger.info(f"Loaded {len(self._data)} measurements from {self._data['profile_index'].nunique()} profiles")
            logger.info(f"Date range: {self._data['date'].min()} to {self._data['date'].max()}")
            logger.info(f"Geographic coverage: Lat {self._data['latitude'].min():.2f} to {self._data['latitude'].max():.2f}")
            logger.info(f"Temperature range: {self._data['temperature'].min():.1f}°C to {self._data['temperature'].max():.1f}°C")
            
            return self._data
            
        except Exception as e:
            logger.error(f"Error loading ARGO data: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate ARGO data"""
        original_count = len(df)
        
        # Remove invalid measurements
        df = df.dropna(subset=['temperature', 'salinity', 'pressure'])
        
        # Filter realistic oceanographic values
        df = df[
            (df['temperature'] >= -5) & (df['temperature'] <= 40) &  # Realistic ocean temperatures
            (df['salinity'] >= 30) & (df['salinity'] <= 40) &        # Realistic ocean salinity
            (df['pressure'] >= 0) & (df['pressure'] <= 3000)         # Realistic ocean depths
        ]
        
        cleaned_count = len(df)
        logger.info(f"Data cleaning: {original_count} → {cleaned_count} measurements ({((cleaned_count/original_count)*100):.1f}% retained)")
        
        return df.reset_index(drop=True)
    
    def get_profiles(self) -> List[ArgoProfile]:
        """Extract individual ARGO profiles from the data"""
        if self._profiles is not None:
            return self._profiles
            
        # Check cache first
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    self._profiles = pickle.load(f)
                logger.info(f"Loaded {len(self._profiles)} profiles from cache")
                return self._profiles
            except Exception:
                logger.warning("Cache file corrupted, reprocessing data")
        
        data = self.load_data()
        profiles = []
        
        # Group by profile_index to create individual profiles
        for profile_id, group in data.groupby('profile_index'):
            # Sort by level_index to ensure proper depth ordering
            group = group.sort_values('level_index')
            
            # Extract profile metadata (same for all levels)
            first_row = group.iloc[0]
            
            # Create measurements list
            measurements = []
            for _, row in group.iterrows():
                measurements.append({
                    "level": int(row['level_index']),
                    "pressure": float(row['pressure']),
                    "temperature": float(row['temperature']),
                    "salinity": float(row['salinity']),
                    "depth_m": float(row['pressure'])  # Approximate depth from pressure
                })
            
            # Create ArgoProfile object
            profile = ArgoProfile(
                profile_id=int(profile_id),
                float_id=f"1900121",  # Float ID from filename
                latitude=float(first_row['latitude']),
                longitude=float(first_row['longitude']),
                date=first_row['date'],
                measurements=measurements
            )
            
            profiles.append(profile)
        
        self._profiles = profiles
        
        # Cache the processed profiles
        with open(self.cache_path, 'wb') as f:
            pickle.dump(profiles, f)
        
        logger.info(f"Processed {len(profiles)} ARGO profiles")
        return profiles
    
    def query_by_location(self, lat_min: float, lat_max: float, 
                         lon_min: float, lon_max: float) -> List[ArgoProfile]:
        """Query profiles by geographic bounds"""
        profiles = self.get_profiles()
        
        filtered = [
            p for p in profiles
            if lat_min <= p.latitude <= lat_max and lon_min <= p.longitude <= lon_max
        ]
        
        logger.info(f"Location query [{lat_min},{lat_max}], [{lon_min},{lon_max}]: {len(filtered)} profiles")
        return filtered
    
    def query_by_date_range(self, start_date: datetime, end_date: datetime) -> List[ArgoProfile]:
        """Query profiles by date range"""
        profiles = self.get_profiles()
        
        filtered = [
            p for p in profiles
            if start_date <= p.date <= end_date
        ]
        
        logger.info(f"Date query {start_date} to {end_date}: {len(filtered)} profiles")
        return filtered
    
    def query_by_temperature_range(self, min_temp: float, max_temp: float) -> List[ArgoProfile]:
        """Query profiles by temperature range"""
        profiles = self.get_profiles()
        
        filtered = []
        for profile in profiles:
            temps = [m["temperature"] for m in profile.measurements]
            if any(min_temp <= temp <= max_temp for temp in temps):
                filtered.append(profile)
        
        logger.info(f"Temperature query {min_temp}°C to {max_temp}°C: {len(filtered)} profiles")
        return filtered
    
    def get_temperature_time_series(self) -> Tuple[List[datetime], List[float]]:
        """Get temperature time series for plotting trends"""
        profiles = self.get_profiles()
        
        dates = []
        surface_temps = []
        
        for profile in profiles:
            if profile.measurements:
                dates.append(profile.date)
                surface_temps.append(profile.measurements[0]["temperature"])
        
        # Sort by date
        sorted_data = sorted(zip(dates, surface_temps))
        dates, surface_temps = zip(*sorted_data) if sorted_data else ([], [])
        
        return list(dates), list(surface_temps)
    
    def get_profile_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the dataset"""
        data = self.load_data()
        profiles = self.get_profiles()
        
        return {
            "total_measurements": len(data),
            "total_profiles": len(profiles),
            "date_range": {
                "start": data['date'].min().isoformat(),
                "end": data['date'].max().isoformat(),
                "span_days": (data['date'].max() - data['date'].min()).days
            },
            "geographic_coverage": {
                "lat_min": float(data['latitude'].min()),
                "lat_max": float(data['latitude'].max()),
                "lon_min": float(data['longitude'].min()),
                "lon_max": float(data['longitude'].max())
            },
            "temperature_stats": {
                "min": float(data['temperature'].min()),
                "max": float(data['temperature'].max()),
                "mean": float(data['temperature'].mean()),
                "std": float(data['temperature'].std())
            },
            "salinity_stats": {
                "min": float(data['salinity'].min()),
                "max": float(data['salinity'].max()),
                "mean": float(data['salinity'].mean()),
                "std": float(data['salinity'].std())
            },
            "depth_stats": {
                "min": float(data['pressure'].min()),
                "max": float(data['pressure'].max()),
                "mean": float(data['pressure'].mean())
            },
            "float_id": "1900121",
            "data_source": "real_argo_csv"
        }
    
    def search_profiles(self, query: str, limit: int = 50) -> List[ArgoProfile]:
        """Search profiles based on text query"""
        profiles = self.get_profiles()
        
        query_lower = query.lower()
        filtered = []
        
        # Simple keyword-based filtering
        if "temperature" in query_lower:
            if "high" in query_lower or "warm" in query_lower:
                filtered = [p for p in profiles if p.measurements and p.measurements[0]["temperature"] > 25]
            elif "low" in query_lower or "cold" in query_lower:
                filtered = [p for p in profiles if p.measurements and p.measurements[0]["temperature"] < 20]
            else:
                filtered = profiles
        elif "salinity" in query_lower:
            if "high" in query_lower:
                filtered = [p for p in profiles if p.measurements and p.measurements[0]["salinity"] > 35]
            elif "low" in query_lower:
                filtered = [p for p in profiles if p.measurements and p.measurements[0]["salinity"] < 35]
            else:
                filtered = profiles
        elif "deep" in query_lower:
            filtered = [p for p in profiles if p.measurements and max(m["pressure"] for m in p.measurements) > 1000]
        elif "shallow" in query_lower:
            filtered = [p for p in profiles if p.measurements and max(m["pressure"] for m in p.measurements) < 200]
        else:
            # Default: return all profiles
            filtered = profiles
        
        return filtered[:limit]
    
    def export_to_netcdf(self, filename: str) -> bool:
        """Export data to NetCDF format (requirement from problem statement)"""
        try:
            import xarray as xr
            
            data = self.load_data()
            
            # Create xarray dataset
            ds = xr.Dataset({
                'temperature': (['measurement'], data['temperature'].values),
                'salinity': (['measurement'], data['salinity'].values),
                'pressure': (['measurement'], data['pressure'].values),
                'latitude': (['measurement'], data['latitude'].values),
                'longitude': (['measurement'], data['longitude'].values),
            }, coords={
                'measurement': range(len(data)),
                'time': (['measurement'], data['date'].values)
            })
            
            # Add attributes
            ds.attrs['title'] = 'ARGO Ocean Profile Data'
            ds.attrs['source'] = 'Float 1900121'
            ds.attrs['creator'] = 'ARGO Ocean Explorer'
            
            ds.to_netcdf(filename)
            logger.info(f"Exported data to NetCDF: {filename}")
            return True
            
        except ImportError:
            logger.warning("xarray not available for NetCDF export")
            return False
        except Exception as e:
            logger.error(f"Error exporting to NetCDF: {e}")
            return False
    
    def export_to_ascii(self, filename: str) -> bool:
        """Export data to ASCII format (requirement from problem statement)"""
        try:
            data = self.load_data()
            
            with open(filename, 'w') as f:
                f.write("# ARGO Ocean Profile Data - Float 1900121\n")
                f.write("# Columns: profile_index, level_index, latitude, longitude, date, pressure, temperature, salinity\n")
                f.write("# Units: degrees, degrees, ISO_datetime, dbar, Celsius, PSU\n")
                f.write("#\n")
                
                for _, row in data.iterrows():
                    f.write(f"{row['profile_index']:3d} {row['level_index']:3d} "
                           f"{row['latitude']:8.3f} {row['longitude']:8.3f} "
                           f"{row['date'].strftime('%Y-%m-%d_%H:%M:%S')} "
                           f"{row['pressure']:8.1f} {row['temperature']:6.2f} {row['salinity']:6.3f}\n")
            
            logger.info(f"Exported data to ASCII: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to ASCII: {e}")
            return False

# Global instance for the real data loader
real_argo_loader = RealArgoDataLoader()