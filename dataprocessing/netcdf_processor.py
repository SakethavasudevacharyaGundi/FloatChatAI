"""
NetCDF processor for ARGO oceanographic data
Handles real ARGO float data from NetCDF files with proper metadata extraction
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone
import re
from config import Config

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
                result = self._extract_argo_data(ds, file_path)
            
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
                'processing_date': datetime.now(timezone.utc).isoformat(),
                'errors': errors
            }
        }
        
        logger.info(f"📊 Combined processing complete: {len(all_profiles)} total profiles from {len(all_metadata)} files")
        return combined_result
    
    def _extract_argo_data(self, ds: xr.Dataset, file_path: Path) -> Dict[str, Any]:
        """Extract data from ARGO NetCDF dataset"""
        
        # Extract metadata
        metadata = self._extract_metadata(ds, file_path)
        
        # Extract profile data
        profiles = self._extract_profiles(ds, metadata)
        
        # Apply quality control
        profiles = self._apply_quality_control(profiles)
        
        # Calculate derived parameters
        profiles = self._calculate_derived_parameters(profiles)
        
        return {
            'profiles': profiles,
            'metadata': metadata,
            'file_info': {
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'processed_at': datetime.now(timezone.utc).isoformat()
            }
        }
    
    def _extract_metadata(self, ds: xr.Dataset, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from ARGO NetCDF dataset"""
        
        metadata = {
            'file_name': file_path.name,
            'format_version': self._get_attr(ds, 'format_version', 'unknown'),
            'data_centre': self._get_attr(ds, 'data_centre', 'unknown'),
            'title': self._get_attr(ds, 'title', 'ARGO Float Data'),
            'institution': self._get_attr(ds, 'institution', 'unknown'),
            'source': self._get_attr(ds, 'source', 'ARGO Float'),
            'references': self._get_attr(ds, 'references', ''),
            'comment': self._get_attr(ds, 'comment', ''),
            'history': self._get_attr(ds, 'history', ''),
            'date_creation': self._get_attr(ds, 'date_creation', ''),
            'date_update': self._get_attr(ds, 'date_update', ''),
            'user_manual_version': self._get_attr(ds, 'user_manual_version', ''),
            'conventions': self._get_attr(ds, 'Conventions', 'CF-1.6')
        }
        
        # Extract float information
        try:
            if 'PLATFORM_NUMBER' in ds.variables:
                platform_numbers = ds['PLATFORM_NUMBER'].values
                if hasattr(platform_numbers[0], 'decode'):
                    # Handle byte strings
                    metadata['platform_number'] = platform_numbers[0].decode('utf-8').strip()
                else:
                    metadata['platform_number'] = str(platform_numbers[0]).strip()
            
            if 'PROJECT_NAME' in ds.variables:
                project_names = ds['PROJECT_NAME'].values
                if hasattr(project_names[0], 'decode'):
                    metadata['project_name'] = project_names[0].decode('utf-8').strip()
                else:
                    metadata['project_name'] = str(project_names[0]).strip()
            
            if 'PI_NAME' in ds.variables:
                pi_names = ds['PI_NAME'].values
                if hasattr(pi_names[0], 'decode'):
                    metadata['pi_name'] = pi_names[0].decode('utf-8').strip()
                else:
                    metadata['pi_name'] = str(pi_names[0]).strip()
            
            # Float deployment information
            if 'LAUNCH_DATE' in ds.variables:
                metadata['launch_date'] = self._decode_argo_time(ds['LAUNCH_DATE'].values[0])
            
            if 'LAUNCH_LATITUDE' in ds.variables:
                metadata['launch_latitude'] = float(ds['LAUNCH_LATITUDE'].values[0])
            
            if 'LAUNCH_LONGITUDE' in ds.variables:
                metadata['launch_longitude'] = float(ds['LAUNCH_LONGITUDE'].values[0])
            
        except Exception as e:
            logger.warning(f"⚠️ Could not extract some metadata: {e}")
        
        # Determine float type (Core/BGC)
        metadata['float_type'] = self._determine_float_type(ds)
        
        # Extract parameter information
        metadata['available_parameters'] = list(self._get_available_parameters(ds).keys())
        
        return metadata
    
    def _extract_profiles(self, ds: xr.Dataset, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract profile data from ARGO dataset"""
        
        profiles = []
        
        try:
            # Get dimensions
            n_prof = ds.dims.get('N_PROF', 1)
            n_levels = ds.dims.get('N_LEVELS', 0)
            
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
                'profile_id': prof_idx,
                'float_id': metadata.get('platform_number', 'unknown'),
                'project_name': metadata.get('project_name', 'unknown'),
                'float_type': metadata.get('float_type', 'core'),
                'measurements': []
            }
            
            # Extract profile-level information
            if 'CYCLE_NUMBER' in ds.variables:
                profile['cycle_number'] = int(ds['CYCLE_NUMBER'][prof_idx].values)
            
            if 'DIRECTION' in ds.variables:
                direction = ds['DIRECTION'][prof_idx].values
                if hasattr(direction, 'decode'):
                    profile['direction'] = direction.decode('utf-8').strip()
                else:
                    profile['direction'] = str(direction).strip()
            
            # Extract profile date/time
            if 'JULD' in ds.variables:
                juld = ds['JULD'][prof_idx].values
                profile['profile_datetime'] = self._decode_argo_time(juld)
            
            # Extract location
            if 'LATITUDE' in ds.variables:
                lat = ds['LATITUDE'][prof_idx].values
                if not np.isnan(lat):
                    profile['lat'] = float(lat)
            
            if 'LONGITUDE' in ds.variables:
                lon = ds['LONGITUDE'][prof_idx].values
                if not np.isnan(lon):
                    profile['lon'] = float(lon)
            
            # Extract measurement data
            for level_idx in range(ds.dims.get('N_LEVELS', 0)):
                measurement = {'level': level_idx}
                
                # Extract pressure/depth (required for valid measurement)
                if 'PRES' in available_params:
                    pres = ds[available_params['PRES']][prof_idx, level_idx].values
                    if not np.isnan(pres) and pres > 0:
                        measurement['pressure_dbar'] = float(pres)
                        # Convert pressure to depth (approximate)
                        measurement['depth_m'] = float(pres * 1.019716)  # Standard conversion
                    else:
                        continue  # Skip invalid pressure measurements
                
                # Extract other parameters
                for argo_param, std_param in available_params.items():
                    if argo_param == 'PRES':
                        continue  # Already handled
                    
                    try:
                        if std_param in ds.variables:
                            value = ds[std_param][prof_idx, level_idx].values
                            if not np.isnan(value):
                                measurement[self.parameter_mappings.get(argo_param, argo_param.lower())] = float(value)
                                
                                # Include quality flag if available
                                qc_var = f"{std_param}_QC"
                                if qc_var in ds.variables:
                                    qc_flag = ds[qc_var][prof_idx, level_idx].values
                                    if hasattr(qc_flag, 'decode'):
                                        qc_flag = qc_flag.decode('utf-8').strip()
                                    measurement[f"{self.parameter_mappings.get(argo_param, argo_param.lower())}_qc"] = str(qc_flag)
                    
                    except Exception as e:
                        logger.debug(f"Could not extract {argo_param} at level {level_idx}: {e}")
                        continue
                
                # Only add measurement if it has depth and at least one other parameter
                if 'depth_m' in measurement and len(measurement) > 2:
                    profile['measurements'].append(measurement)
            
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
            reference_date = datetime(1950, 1, 1, tzinfo=timezone.utc)
            
            # Convert Julian day to datetime
            dt = reference_date + pd.Timedelta(days=float(julian_day))
            
            return dt.isoformat()
            
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