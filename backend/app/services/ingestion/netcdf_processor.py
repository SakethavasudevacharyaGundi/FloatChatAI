"""
NetCDF processor for ARGO oceanographic data
Handles real ARGO float data from NetCDF files with proper metadata extraction
"""

import xarray as xr
import asyncpg
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone
import re
from app.core.config import config
from app.services.data.bgc_generator import bgc_generator
from datetime import datetime, timezone, timedelta
logger = logging.getLogger(__name__)

class NetCDFProcessor:
    """Process ARGO NetCDF files and extract oceanographic data"""
    
    def __init__(self):
        self.supported_formats = ['.nc', '.netcdf']
        self.quality_flags = {
            'good': [1, 2],  # Good and probably good data
            'bad': [3, 4, 8, 9],  # Bad, missing, or interpolated data
            'questionable': [0, 5, 6, 7]  # No QC, changed, or corrected data
        }
        
        # Standard ARGO parameter mappings
        self.parameter_mappings = {
            'TEMP': 'temperature_c',
            'PSAL': 'salinity_psu', 
            'PRES': 'pressure_dbar',
            'DOXY': 'oxygen_umol_kg',
            'CHLA': 'chlorophyll_mg_m3',
            'NITRATE': 'nitrate_umol_kg',
            'PH_IN_SITU_TOTAL': 'ph_total',
            'BBP700': 'backscatter_700nm',
            'CDOM': 'cdom_ppb'
        }
        
        logger.info("🌊 NetCDF Processor initialized for ARGO data")
    
    def process_argo_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single ARGO NetCDF file"""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format. Expected: {self.supported_formats}")
        
        try:
            logger.info(f"📂 Processing ARGO file: {file_path.name}")
            
            # Load NetCDF dataset
            with xr.open_dataset(file_path, decode_times=False) as ds:

                metadata = self._extract_metadata(ds)

                profiles = self._extract_profiles(
                    ds,
                    metadata
                )

                result = {
                    "metadata": metadata,
                    "profiles": profiles
                }
                
                logger.info(f"✅ Successfully processed {len(result.get('profiles', []))} profiles from {file_path.name}")
                return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            raise
    
    def process_multiple_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """Process multiple ARGO NetCDF files and combine data"""
        
        all_profiles = []
        all_metadata = []
        errors = []
        
        logger.info(f"📁 Processing {len(file_paths)} ARGO files")
        
        for file_path in file_paths:
            try:
                result = self.process_argo_file(file_path)
                all_profiles.extend(result.get('profiles', []))
                if result.get('metadata'):
                    all_metadata.append(result['metadata'])
                    
            except Exception as e:
                error_msg = f"Failed to process {Path(file_path).name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"⚠️ {error_msg}")
        
        # Combine into final result
        combined_result = {
            'profiles': all_profiles,
            'metadata': all_metadata,
            'summary': {
                'total_files': len(file_paths),
                'successful_files': len(all_metadata),
                'failed_files': len(errors),
                'total_profiles': len(all_profiles),
                'errors': errors
            }
        }
        
        logger.info(f"📊 Combined processing complete: {len(all_profiles)} total profiles from {len(all_metadata)} files")
        return combined_result
    
    def _extract_metadata(self, ds):

        metadata = {}

        try:

            # Platform number
            if 'platform_number' in ds.variables:

                raw = ds['platform_number'].values

                try:

                    metadata['platform_number'] = ''.join(
                        x.decode('utf-8') if isinstance(x, bytes) else str(x)
                        for x in raw.flatten()
                    ).strip()

                except:

                    metadata['platform_number'] = str(raw)


            # Cycle number
            if 'cycle_number' in ds.variables:

                metadata['cycle_number'] = int(
                    ds['cycle_number'].values.flatten()[0]
                )


            # Latitude
            if 'latitude' in ds.variables:

                metadata['latitude'] = float(
                    ds['latitude'].values.flatten()[0]
                )


            # Longitude
            if 'longitude' in ds.variables:

                metadata['longitude'] = float(
                    ds['longitude'].values.flatten()[0]
                )


            # JULD → datetime
            if 'juld' in ds.variables:

                juld = float(
                    ds['juld'].values.flatten()[0]
                )

                reference_date = datetime(
                    1950, 1, 1
                )

                metadata['profile_datetime'] = (
                    reference_date +
                    timedelta(days=juld)
                )


            # Project name
            if 'project_name' in ds.variables:

                raw = ds['project_name'].values

                try:

                    metadata['project_name'] = ''.join(
                        x.decode('utf-8') if isinstance(x, bytes) else str(x)
                        for x in raw.flatten()
                    ).strip()

                except:

                    metadata['project_name'] = str(raw)


            # Float type
            metadata['float_type'] = 'argo'


        except Exception as e:

            logger.warning(
                f"Metadata extraction failed: {e}"
            )

        return metadata

    
    def _extract_profiles(self, ds: xr.Dataset, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract profile data from ARGO dataset"""
        
        profiles = []
        
        try:
            # Get dimensions
            # Debug dataset structure
            print("\n===================")
            print("DATASET STRUCTURE")
            print("===================")

            print("\nDimensions:")
            print(ds.sizes)

            print("\nVariables:")
            print(list(ds.variables.keys()))

            print("\nCoordinates:")
            print(list(ds.coords.keys()))

            # Get dimensions safely
            n_prof = (
                ds.sizes.get('N_PROF')
                or ds.sizes.get('n_prof')
                or 1
            )

            n_levels = (
                ds.sizes.get('N_LEVELS')
                or ds.sizes.get('n_levels')
                or ds.sizes.get('N_LEVEL')
                or ds.sizes.get('n_level')
                or ds.sizes.get('DEPTH')
                or 0
            )

            print(f"\nN_PROF: {n_prof}")
            print(f"N_LEVELS: {n_levels}")
            
            if n_levels == 0:
                logger.warning("⚠️ No measurement levels found in dataset")
                return profiles
            
            # Get available parameters
            available_params = self._get_available_parameters(ds)
            
            for prof_idx in range(n_prof):
                try:
                    profile = self._extract_single_profile(ds, prof_idx, available_params, metadata)
                    if profile and len(profile.get('measurements', [])) > 0:
                        profiles.append(profile)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error extracting profile {prof_idx}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"❌ Error extracting profiles: {e}")
        
        return profiles
    
    def _extract_single_profile(self, ds: xr.Dataset, prof_idx: int, 
                               available_params: Dict[str, str], metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract a single profile from ARGO dataset"""
        
        try:
            profile = {

                'profile_id': (
                    f"{metadata.get('platform_number')}_"
                    f"{metadata.get('cycle_number')}"
                ),

                'float_id': metadata.get(
                    'platform_number'
                ),

                'cycle_number': metadata.get(
                    'cycle_number'
                ),

                'profile_datetime': metadata.get(
                    'profile_datetime'
                ),

                'lat': metadata.get(
                    'latitude'
                ),

                'lon': metadata.get(
                    'longitude'
                ),

                'project_name': metadata.get(
                    'project_name',
                    'unknown'
                ),

                'float_type': metadata.get(
                    'float_type',
                    'argo'
                ),

                'measurements': []
            }
            
            
            # Extract profile-level information
            if 'cycle_number' in ds.variables:
                profile['cycle_number'] = int(ds['cycle_number'][prof_idx].values)
            
            if 'DIRECTION' in ds.variables:
                direction = ds['DIRECTION'][prof_idx].values
                if hasattr(direction, 'decode'):
                    profile['direction'] = direction.decode('utf-8').strip()
                else:
                    profile['direction'] = str(direction).strip()
            
            # Extract profile date/time
            if 'juld' in ds.variables:
                juld = ds['juld'][prof_idx].values
                profile['profile_datetime'] = self._decode_argo_time(juld)
            
            # Extract location
            if 'latitude' in ds.variables:
                lat = ds['latitude'][prof_idx].values
                if not np.isnan(lat):
                    profile['lat'] = float(lat)
            
            if 'longitude' in ds.variables:
                lon = ds['longitude'][prof_idx].values
                if not np.isnan(lon):
                    profile['lon'] = float(lon)
            
            # Extract measurement data
            n_levels = (
                ds.sizes.get('N_LEVELS')
                or ds.sizes.get('n_levels')
                or 0
            )

            for level_idx in range(n_levels):
                measurement = {'level': level_idx}
                print(f"\nProcessing level: {level_idx}")
                
                # Extract pressure/depth (required for valid measurement)
# Extract pressure/depth (required for valid measurement)

                if 'pres' in ds.variables:

                    pres = ds['pres'][prof_idx, level_idx].values

                    try:

                        pres = float(pres)

                    except:

                        continue

                    if not np.isnan(pres) and pres > 0:

                        measurement['pressure_dbar'] = pres

                        measurement['depth_m'] = pres * 1.019716

                    else:

                        continue
                # Extract other parameters
                    # Temperature
                    if 'temp' in ds.variables:

                        temp = ds['temp'][prof_idx, level_idx].values

                        try:

                            temp = float(temp)

                            if not np.isnan(temp):

                                measurement['temperature_c'] = temp

                        except:
                            pass


                    # Salinity
                    if 'psal' in ds.variables:

                        psal = ds['psal'][prof_idx, level_idx].values

                        try:

                            psal = float(psal)

                            if not np.isnan(psal):

                                measurement['salinity_psu'] = psal
                                bgc = bgc_generator.generate(
                                    temperature=temp,
                                    salinity=psal,
                                    pressure=pres
                                )

                                measurement.update(bgc)


                        except:
                            pass
                    if 'depth_m' in measurement and len(measurement) > 2:
                        profile['measurements'].append(measurement)
                # Only add measurement if it has depth and at least one other parameter

            # Add summary statistics
            if profile['measurements']:
                depths = [m['depth_m'] for m in profile['measurements']]
                profile['max_depth'] = max(depths)
                profile['min_depth'] = min(depths)
                profile['n_levels'] = len(profile['measurements'])

            return profile if profile['measurements'] else None

        except Exception as e:
            logger.warning(f"⚠️ Error extracting profile {prof_idx}: {e}")
            return None
        
    def _get_available_parameters(self, ds: xr.Dataset) -> Dict[str, str]:
        """Get available ARGO parameters in the dataset"""
        
        available = {}
        
        # Check for core parameters
        for argo_param, variable_names in {
            'PRES': ['PRES', 'PRES_ADJUSTED'],
            'TEMP': ['TEMP', 'TEMP_ADJUSTED'],
            'PSAL': ['PSAL', 'PSAL_ADJUSTED']
        }.items():
            for var_name in variable_names:
                if var_name in ds.variables:
                    available[argo_param] = var_name
                    break
        
        # Check for BGC parameters
        for argo_param, variable_names in {
            'DOXY': ['DOXY', 'DOXY_ADJUSTED'],
            'CHLA': ['CHLA', 'CHLA_ADJUSTED'],
            'NITRATE': ['NITRATE', 'NITRATE_ADJUSTED'],
            'PH_IN_SITU_TOTAL': ['PH_IN_SITU_TOTAL', 'PH_IN_SITU_TOTAL_ADJUSTED'],
            'BBP700': ['BBP700', 'BBP700_ADJUSTED'],
            'CDOM': ['CDOM', 'CDOM_ADJUSTED']
        }.items():
            for var_name in variable_names:
                if var_name in ds.variables:
                    available[argo_param] = var_name
                    break
        
        return available
    
    def _determine_float_type(self, ds: xr.Dataset) -> str:
        """Determine if float is Core-only or BGC float"""
        
        bgc_parameters = ['DOXY', 'CHLA', 'NITRATE', 'PH_IN_SITU_TOTAL', 'BBP700', 'CDOM']
        
        for param in bgc_parameters:
            if param in ds.variables or f"{param}_ADJUSTED" in ds.variables:
                return 'bgc'
        
        return 'core'
    
    def _apply_quality_control(self, profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply quality control to profile data"""
        
        qc_profiles = []
        
        for profile in profiles:
            qc_profile = profile.copy()
            qc_measurements = []
            
            for measurement in profile['measurements']:
                qc_measurement = measurement.copy()
                
                # Check quality flags and remove bad data
                remove_measurement = False
                
                for param, value in measurement.items():
                    if param.endswith('_qc'):
                        qc_flag = str(value).strip()
                        
                        # Remove measurements with bad quality flags
                        if qc_flag in ['3', '4', '8', '9']:  # Bad data flags
                            remove_measurement = True
                            break
                
                # Additional range checks
                if not remove_measurement:
                    # Temperature range check
                    if 'temperature_c' in measurement:
                        temp = measurement['temperature_c']
                        if temp < -2.5 or temp > 40:  # Realistic ocean temperature range
                            remove_measurement = True
                    
                    # Salinity range check
                    if 'salinity_psu' in measurement:
                        sal = measurement['salinity_psu']
                        if sal < 0 or sal > 50:  # Realistic salinity range
                            remove_measurement = True
                    
                    # Depth check
                    if 'depth_m' in measurement:
                        depth = measurement['depth_m']
                        if depth < 0 or depth > 6000:  # Reasonable depth range for ARGO
                            remove_measurement = True
                
                if not remove_measurement:
                    qc_measurements.append(qc_measurement)
            
            if qc_measurements:
                qc_profile['measurements'] = qc_measurements
                qc_profile['n_levels'] = len(qc_measurements)
                qc_profiles.append(qc_profile)
        
        removed_profiles = len(profiles) - len(qc_profiles)
        if removed_profiles > 0:
            logger.info(f"🔍 Quality control removed {removed_profiles} profiles")
        
        return qc_profiles
    
    def _calculate_derived_parameters(self, profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate derived oceanographic parameters"""
        
        for profile in profiles:
            for measurement in profile['measurements']:
                # Calculate potential temperature if temperature and pressure available
                if 'temperature_c' in measurement and 'pressure_dbar' in measurement:
                    # Simplified potential temperature calculation
                    temp = measurement['temperature_c']
                    pres = measurement['pressure_dbar']
                    # Approximate potential temperature (simplified formula)
                    pot_temp = temp - (pres * 0.0003)  # Very simplified
                    measurement['potential_temperature_c'] = pot_temp
                
                # Calculate density if temperature, salinity, and pressure available
                if all(param in measurement for param in ['temperature_c', 'salinity_psu', 'pressure_dbar']):
                    density = self._calculate_seawater_density(
                        measurement['temperature_c'],
                        measurement['salinity_psu'],
                        measurement['pressure_dbar']
                    )
                    measurement['density_kg_m3'] = density
                
                # Add geographical region if coordinates available
                if 'lat' in profile and 'lon' in profile:
                    region = self._determine_ocean_region(profile['lat'], profile['lon'])
                    measurement['region'] = region
                    profile['region'] = region
        
        return profiles
    
    def _calculate_seawater_density(self, temperature: float, salinity: float, pressure: float) -> float:
        """Calculate seawater density using simplified equation of state"""
        
        # Simplified UNESCO equation of state for seawater density
        # This is a very simplified version - real applications should use TEOS-10
        
        # Convert pressure from dbar to MPa
        p = pressure / 1000.0
        
        # Temperature and salinity effects on density
        rho0 = 1024.5  # Reference density
        
        # Temperature effect (simplified)
        alpha = 2e-4  # Thermal expansion coefficient
        temp_effect = -alpha * (temperature - 15.0)
        
        # Salinity effect (simplified)
        beta = 7.8e-4  # Haline contraction coefficient
        sal_effect = beta * (salinity - 35.0)
        
        # Pressure effect (simplified compressibility)
        gamma = 4.4e-10  # Compressibility
        pres_effect = gamma * pressure
        
        density = rho0 * (1 + sal_effect + temp_effect + pres_effect)
        
        return density
    
    def _determine_ocean_region(self, lat: float, lon: float) -> str:
        """Determine ocean region based on coordinates"""
        
        # Simplified regional classification focused on Indian Ocean
        if -90 <= lat <= 90 and 20 <= lon <= 150:
            if lat >= 0:
                if lon < 70:
                    return "Arabian Sea"
                elif lon < 100:
                    return "Bay of Bengal"
                else:
                    return "Southeast Asian Seas"
            else:  # Southern hemisphere
                if lon < 60:
                    return "Southwest Indian Ocean"
                elif lon < 110:
                    return "Central Indian Ocean"
                else:
                    return "Southeast Indian Ocean"
        
        # Other oceans (simplified)
        elif -180 <= lon < 20 or 150 <= lon <= 180:
            if lat > 0:
                return "North Atlantic/Pacific"
            else:
                return "South Atlantic/Pacific"
        else:
            return "Unknown Region"
    
    def _decode_argo_time(self, julian_day: Union[float, np.ndarray]) -> Optional[str]:
        """Convert ARGO Julian day to ISO datetime string"""
        
        try:
            if np.isnan(julian_day):
                return None
            
            # ARGO reference date is 1950-01-01
            reference_date = datetime(1950, 1, 1)
            
            # Convert Julian day to datetime
            dt = reference_date + pd.Timedelta(days=float(julian_day))
            
            return dt
            
        except Exception as e:
            logger.warning(f"⚠️ Could not decode ARGO time {julian_day}: {e}")
            return None
    
    def _get_attr(self, ds: xr.Dataset, attr_name: str, default: Any = None) -> Any:
        """Safely get dataset attribute"""
        
        try:
            if attr_name in ds.attrs:
                value = ds.attrs[attr_name]
                if hasattr(value, 'decode'):
                    return value.decode('utf-8').strip()
                return str(value).strip()
        except Exception:
            pass
        
        return default
    
    def convert_to_dataframe(self, processed_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert processed ARGO data to pandas DataFrame"""
        
        if not processed_data.get('profiles'):
            return pd.DataFrame()
        
        all_measurements = []
        
        for profile in processed_data['profiles']:
            profile_info = {
                'profile_id': profile.get('profile_id'),
                'float_id': profile.get('float_id'),
                'cycle_number': profile.get('cycle_number'),
                'profile_datetime': profile.get('profile_datetime'),
                'lat': profile.get('lat'),
                'lon': profile.get('lon'),
                'direction': profile.get('direction'),
                'float_type': profile.get('float_type'),
                'region': profile.get('region'),
                'max_depth': profile.get('max_depth'),
                'n_levels': profile.get('n_levels')
            }
            
            for measurement in profile.get('measurements', []):
                # Combine profile info with measurement
                row = {**profile_info, **measurement}
                all_measurements.append(row)
        
        df = pd.DataFrame(all_measurements)
        
        # Convert datetime strings to pandas datetime
        if 'profile_datetime' in df.columns:
            df['profile_datetime'] = pd.to_datetime(df['profile_datetime'], errors='coerce')
        
        # Sort by float, cycle, and depth
        sort_cols = ['float_id', 'cycle_number', 'depth_m']
        sort_cols = [col for col in sort_cols if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        
        logger.info(f"📊 Converted to DataFrame: {len(df)} measurements from {df['float_id'].nunique() if 'float_id' in df else 0} floats")
        
        return df
    
    def get_data_summary(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for processed ARGO data"""
        
        if not processed_data.get('profiles'):
            return {'error': 'No profile data available'}
        
        df = self.convert_to_dataframe(processed_data)
        
        if df.empty:
            return {'error': 'No measurement data available'}
        
        summary = {
            'total_profiles': len(processed_data['profiles']),
            'total_measurements': len(df),
            'unique_floats': df['float_id'].nunique() if 'float_id' in df else 0,
            'date_range': {
                'start': df['profile_datetime'].min().isoformat() if 'profile_datetime' in df and df['profile_datetime'].notna().any() else None,
                'end': df['profile_datetime'].max().isoformat() if 'profile_datetime' in df and df['profile_datetime'].notna().any() else None
            },
            'geographic_bounds': {
                'min_lat': df['lat'].min() if 'lat' in df else None,
                'max_lat': df['lat'].max() if 'lat' in df else None,
                'min_lon': df['lon'].min() if 'lon' in df else None,
                'max_lon': df['lon'].max() if 'lon' in df else None
            },
            'depth_range': {
                'min_depth': df['depth_m'].min() if 'depth_m' in df else None,
                'max_depth': df['depth_m'].max() if 'depth_m' in df else None
            },
            'available_parameters': [col for col in df.columns if col.endswith(('_c', '_psu', '_dbar', '_kg', '_m3', '_umol_kg'))],
            'float_types': df['float_type'].value_counts().to_dict() if 'float_type' in df else {},
            'regions': df['region'].value_counts().to_dict() if 'region' in df else {},
            'processing_summary': processed_data.get('summary', {})
        }
        
        return summary
    async def insert_into_postgres(
    self,
    processed_data: Dict[str, Any]) -> None:
        """Insert processed ARGO data into PostgreSQL database"""
        conn = await asyncpg.connect(
            config.DATABASE_URL
    )

        try:

            profiles = processed_data.get("profiles", [])

            print(f"Inserting {len(profiles)} profiles...")

            for profile in profiles:

                float_id = profile.get("float_id", "unknown")

                # Insert float
                await conn.execute(
                    """
                    INSERT INTO floats (
                        float_id,
                        region,
                        status
                    )
                    VALUES ($1,$2,$3)
                    ON CONFLICT (float_id) DO NOTHING
                    """,
                    float_id,
                    profile.get("region", "unknown"),
                    "active"
                )

                profile_key = f"{float_id}_{profile.get('cycle_number', 0)}"

                # Insert profile
                await conn.execute(
                    """
                    INSERT INTO profiles (
                        profile_id,
                        float_id,
                        cycle_number,
                        profile_datetime,
                        lat,
                        lon,
                        n_levels
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7)
                    ON CONFLICT (profile_id) DO NOTHING
                    """,
                    profile_key,
                    float_id,
                    profile.get("cycle_number", 0),
                    profile.get("profile_datetime"),
                    profile.get("lat"),
                    profile.get("lon"),
                    profile.get("n_levels", 0)
                )

                # Insert measurements
                for m in profile.get("measurements", []):

                    await conn.execute(
                        """
                        INSERT INTO measurements (
                            profile_id,
                            depth_m,
                            pressure_dbar,
                            temperature_c,
                            salinity_psu,
                            oxygen_umol_kg,
                            chlorophyll_mg_m3,
                            nitrate_umol_kg,
                            ph_total
                        )
                        VALUES (
                            $1,$2,$3,$4,$5,$6,$7,$8,$9
                        )
                        """,
                        profile_key,
                        m.get("depth_m"),
                        m.get("pressure_dbar"),
                        m.get("temperature_c"),
                        m.get("salinity_psu"),
                        m.get("oxygen_umol_kg"),
                        m.get("chlorophyll_mg_m3"),
                        m.get("nitrate_umol_kg"),
                        m.get("ph_total")
                    )

            print("✅ PostgreSQL ingestion complete")

        finally:

            await conn.close()