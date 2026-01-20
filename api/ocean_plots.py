"""
Advanced ocean visualization engine with Plotly and Folium
Supports depth profiles, trajectories, heatmaps, and interactive maps
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import folium
from folium import plugins
import logging
from config import Config

logger = logging.getLogger(__name__)

class OceanVisualizationEngine:
    """Advanced visualization engine for ARGO oceanographic data"""
    
    def __init__(self):
        self.color_scales = {
            'temperature': 'RdYlBu_r',  # Red-Yellow-Blue for temperature
            'salinity': 'Viridis',      # Green-Blue for salinity
            'oxygen': 'Blues',          # Blue for oxygen
            'chlorophyll': 'Greens',    # Green for chlorophyll
            'depth': 'Deep',            # Deep blue for depth
            'ph': 'RdYlGn',            # Red-Yellow-Green for pH
            'nitrate': 'Plasma'         # Purple-Pink for nitrate
        }
        
        self.default_layout = {
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'paper_bgcolor': 'white',
            'plot_bgcolor': 'white',
            'margin': dict(l=60, r=60, t=60, b=60)
        }
        
        logger.info("🎨 Ocean Visualization Engine initialized")
    
    def create_visualization(self, data: Union[pd.DataFrame, List[Dict]], viz_type: str, **kwargs) -> Optional[go.Figure]:
        """Create visualization based on type and data"""
        
        try:
            # Convert data to DataFrame if needed
            if isinstance(data, list):
                if not data:
                    return self._create_empty_plot("No data available")
                df = pd.DataFrame(data)
            else:
                df = data.copy() if not data.empty else pd.DataFrame()
            
            if df.empty:
                return self._create_empty_plot("No data available for visualization")
            
            # Visualization methods mapping
            viz_methods = {
                'depth_profile': self.create_depth_profile,
                'trajectory_map': self.create_trajectory_map_plotly,
                'time_series': self.create_time_series,
                'scatter_plot': self.create_scatter_plot,
                'heatmap': self.create_parameter_heatmap,
                'comparison_chart': self.create_comparison_chart,
                'bgc_profile': self.create_bgc_profile,
                'regional_summary': self.create_regional_summary,
                'correlation_matrix': self.create_correlation_matrix
            }
            
            viz_method = viz_methods.get(viz_type, self.create_default_plot)
            
            fig = viz_method(df, **kwargs)
            
            # Apply consistent styling
            if fig:
                fig.update_layout(**self.default_layout)
                fig.update_layout(height=Config.PLOT_HEIGHT, width=Config.PLOT_WIDTH)
            
            logger.info(f"✅ Created {viz_type} visualization with {len(df)} data points")
            return fig
            
        except Exception as e:
            logger.error(f"❌ Error creating {viz_type} visualization: {e}")
            return self._create_error_plot(f"Error creating {viz_type}: {str(e)}")
    
    def create_depth_profile(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create temperature/salinity vs depth profiles"""
        
        # Check required columns
        required_cols = ['depth_m']
        if not all(col in df.columns for col in required_cols):
            return self._create_error_plot("Depth data required for depth profile")
        
        # Create subplots
        has_temp = 'temperature_c' in df.columns and df['temperature_c'].notna().any()
        has_sal = 'salinity_psu' in df.columns and df['salinity_psu'].notna().any()
        
        if not has_temp and not has_sal:
            return self._create_error_plot("Temperature or salinity data required")
        
        if has_temp and has_sal:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Temperature Profile', 'Salinity Profile'),
                shared_yaxes=True,
                horizontal_spacing=0.15
            )
        else:
            fig = go.Figure()
        
        # Clean data
        clean_df = df.dropna(subset=['depth_m'])
        clean_df = clean_df.sort_values('depth_m')
        
        # Temperature profile
        if has_temp:
            temp_data = clean_df.dropna(subset=['temperature_c'])
            if not temp_data.empty:
                if has_temp and has_sal:
                    fig.add_trace(
                        go.Scatter(
                            x=temp_data['temperature_c'],
                            y=-temp_data['depth_m'],  # Negative for ocean depth convention
                            mode='lines+markers',
                            name='Temperature',
                            line=dict(color='red', width=3),
                            marker=dict(size=6, color='red')
                        ),
                        row=1, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=temp_data['temperature_c'],
                            y=-temp_data['depth_m'],
                            mode='lines+markers',
                            name='Temperature (°C)',
                            line=dict(color='red', width=3),
                            marker=dict(size=6, color='red')
                        )
                    )
        
        # Salinity profile
        if has_sal:
            sal_data = clean_df.dropna(subset=['salinity_psu'])
            if not sal_data.empty:
                if has_temp and has_sal:
                    fig.add_trace(
                        go.Scatter(
                            x=sal_data['salinity_psu'],
                            y=-sal_data['depth_m'],
                            mode='lines+markers',
                            name='Salinity',
                            line=dict(color='blue', width=3),
                            marker=dict(size=6, color='blue')
                        ),
                        row=1, col=2
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=sal_data['salinity_psu'],
                            y=-sal_data['depth_m'],
                            mode='lines+markers',
                            name='Salinity (PSU)',
                            line=dict(color='blue', width=3),
                            marker=dict(size=6, color='blue')
                        )
                    )
        
        # Update layout
        if has_temp and has_sal:
            fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
            fig.update_yaxes(title_text="Depth (m)", row=1, col=1)
            title = "Ocean Depth Profiles: Temperature & Salinity"
        elif has_temp:
            fig.update_xaxes(title_text="Temperature (°C)")
            fig.update_yaxes(title_text="Depth (m)")
            title = "Ocean Temperature Depth Profile"
        else:
            fig.update_xaxes(title_text="Salinity (PSU)")
            fig.update_yaxes(title_text="Depth (m)")
            title = "Ocean Salinity Depth Profile"
        
        fig.update_layout(
            title=title,
            showlegend=False if has_temp and has_sal else True,
            hovermode='closest'
        )
        
        return fig
    
    def create_trajectory_map_plotly(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create float trajectory map using Plotly"""
        
        if 'lat' not in df.columns or 'lon' not in df.columns:
            return self._create_error_plot("Latitude/Longitude required for trajectory map")
        
        # Clean coordinate data
        map_data = df.dropna(subset=['lat', 'lon'])
        map_data = map_data[(map_data['lat'].between(-90, 90)) & (map_data['lon'].between(-180, 180))]
        
        if map_data.empty:
            return self._create_error_plot("No valid coordinate data available")
        
        fig = go.Figure()
        
        # Group by float_id if available
        if 'float_id' in map_data.columns:
            colors = px.colors.qualitative.Set1
            for i, (float_id, float_data) in enumerate(map_data.groupby('float_id')):
                # Sort by time if available
                if 'profile_datetime' in float_data.columns:
                    float_data = float_data.sort_values('profile_datetime')
                
                color = colors[i % len(colors)]
                
                # Add trajectory line
                fig.add_trace(go.Scattermapbox(
                    lat=float_data['lat'],
                    lon=float_data['lon'],
                    mode='lines+markers',
                    name=f'Float {float_id}',
                    line=dict(width=3, color=color),
                    marker=dict(size=8, color=color),
                    hovertemplate=(
                        '<b>Float %{text}</b><br>'
                        'Lat: %{lat:.3f}°<br>'
                        'Lon: %{lon:.3f}°<br>'
                        '<extra></extra>'
                    ),
                    text=[str(float_id)] * len(float_data)
                ))
        else:
            # Single trajectory
            fig.add_trace(go.Scattermapbox(
                lat=map_data['lat'],
                lon=map_data['lon'],
                mode='lines+markers',
                name='ARGO Float Trajectory',
                line=dict(width=3, color='blue'),
                marker=dict(size=8, color='red'),
                hovertemplate=(
                    'Lat: %{lat:.3f}°<br>'
                    'Lon: %{lon:.3f}°<br>'
                    '<extra></extra>'
                )
            ))
        
        # Calculate map center and zoom
        center_lat = map_data['lat'].mean()
        center_lon = map_data['lon'].mean()
        
        # Calculate zoom based on data spread
        lat_range = map_data['lat'].max() - map_data['lat'].min()
        lon_range = map_data['lon'].max() - map_data['lon'].min()
        max_range = max(lat_range, lon_range)
        
        if max_range > 50:
            zoom = 1
        elif max_range > 20:
            zoom = 2
        elif max_range > 10:
            zoom = 3
        else:
            zoom = 4
        
        fig.update_layout(
            title="🌊 ARGO Float Trajectories",
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=600
        )
        
        return fig
    
    def create_time_series(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create time series plot for ocean parameters"""
        
        if 'profile_datetime' not in df.columns:
            return self._create_error_plot("DateTime data required for time series")
        
        # Convert datetime
        time_data = df.copy()
        time_data['profile_datetime'] = pd.to_datetime(time_data['profile_datetime'], errors='coerce')
        time_data = time_data.dropna(subset=['profile_datetime'])
        time_data = time_data.sort_values('profile_datetime')
        
        if time_data.empty:
            return self._create_error_plot("No valid datetime data available")
        
        fig = go.Figure()
        
        # Plot multiple parameters
        params_to_plot = []
        if 'temperature_c' in time_data.columns and time_data['temperature_c'].notna().any():
            params_to_plot.append(('temperature_c', 'Temperature (°C)', 'red'))
        if 'salinity_psu' in time_data.columns and time_data['salinity_psu'].notna().any():
            params_to_plot.append(('salinity_psu', 'Salinity (PSU)', 'blue'))
        if 'oxygen_umol_kg' in time_data.columns and time_data['oxygen_umol_kg'].notna().any():
            params_to_plot.append(('oxygen_umol_kg', 'Oxygen (μmol/kg)', 'green'))
        
        if not params_to_plot:
            return self._create_error_plot("No plottable parameters found")
        
        # Create dual y-axis if multiple parameters
        if len(params_to_plot) > 1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for i, (param, label, color) in enumerate(params_to_plot):
            param_data = time_data.dropna(subset=[param])
            
            if not param_data.empty:
                secondary_y = i > 0 and len(params_to_plot) > 1
                
                fig.add_trace(
                    go.Scatter(
                        x=param_data['profile_datetime'],
                        y=param_data[param],
                        mode='lines+markers',
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=6)
                    ),
                    secondary_y=secondary_y
                )
        
        # Update layout
        fig.update_xaxes(title_text="Date")
        
        if len(params_to_plot) > 1:
            # Dual y-axis
            fig.update_yaxes(title_text=params_to_plot[0][1], secondary_y=False)
            fig.update_yaxes(title_text=params_to_plot[1][1], secondary_y=True)
        else:
            fig.update_yaxes(title_text=params_to_plot[0][1])
        
        fig.update_layout(
            title="📈 Ocean Parameters Time Series",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create temperature vs salinity scatter plot (T-S diagram)"""
        
        if 'temperature_c' not in df.columns or 'salinity_psu' not in df.columns:
            return self._create_error_plot("Temperature and Salinity data required for T-S diagram")
        
        # Clean data
        ts_data = df.dropna(subset=['temperature_c', 'salinity_psu'])
        
        if ts_data.empty:
            return self._create_error_plot("No valid Temperature-Salinity pairs available")
        
        # Color by depth if available
        if 'depth_m' in ts_data.columns and ts_data['depth_m'].notna().any():
            color_col = 'depth_m'
            color_scale = 'Deep'
            hover_data = ['depth_m']
        else:
            color_col = None
            color_scale = None
            hover_data = None
        
        fig = px.scatter(
            ts_data,
            x='salinity_psu',
            y='temperature_c',
            color=color_col,
            color_continuous_scale=color_scale,
            title='🌡️ Temperature-Salinity Diagram',
            labels={
                'salinity_psu': 'Salinity (PSU)',
                'temperature_c': 'Temperature (°C)',
                'depth_m': 'Depth (m)'
            },
            hover_data=hover_data
        )
        
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
        
        # Add density contours if enough data
        if len(ts_data) > 50:
            try:
                fig.add_trace(
                    go.Histogram2dContour(
                        x=ts_data['salinity_psu'],
                        y=ts_data['temperature_c'],
                        name='Density',
                        showscale=False,
                        line=dict(color='gray', width=1),
                        opacity=0.3
                    )
                )
            except:
                pass  # Skip contours if they fail
        
        return fig
    
    def create_parameter_heatmap(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create geographic heatmap of ocean parameters"""
        
        if 'lat' not in df.columns or 'lon' not in df.columns:
            return self._create_error_plot("Coordinates required for heatmap")
        
        # Determine parameter to plot
        parameter = kwargs.get('parameter', 'temperature_c')
        
        # Check available parameters
        available_params = [col for col in df.columns 
                          if col.endswith(('_c', '_psu', '_m', '_kg', '_m3')) and df[col].notna().any()]
        
        if parameter not in df.columns or not df[parameter].notna().any():
            if available_params:
                parameter = available_params[0]
            else:
                return self._create_error_plot("No plottable parameters found")
        
        # Clean data
        heatmap_data = df.dropna(subset=['lat', 'lon', parameter])
        heatmap_data = heatmap_data[(heatmap_data['lat'].between(-90, 90)) & 
                                   (heatmap_data['lon'].between(-180, 180))]
        
        if heatmap_data.empty:
            return self._create_error_plot(f"No valid data for {parameter}")
        
        # Get color scale
        param_name = parameter.split('_')[0]
        color_scale = self.color_scales.get(param_name, 'Viridis')
        
        # Create density heatmap
        fig = px.density_mapbox(
            heatmap_data,
            lat='lat',
            lon='lon',
            z=parameter,
            radius=15,
            center=dict(
                lat=heatmap_data['lat'].mean(),
                lon=heatmap_data['lon'].mean()
            ),
            zoom=3,
            mapbox_style="open-street-map",
            title=f"🗺️ {parameter.replace('_', ' ').title()} Distribution",
            color_continuous_scale=color_scale,
            opacity=0.7
        )
        
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
        
        return fig
    
    def create_bgc_profile(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create BGC (biogeochemical) parameters depth profile"""
        
        bgc_params = {
            'oxygen_umol_kg': ('Oxygen (μmol/kg)', 'blue'),
            'chlorophyll_mg_m3': ('Chlorophyll (mg/m³)', 'green'),
            'nitrate_umol_kg': ('Nitrate (μmol/kg)', 'orange'),
            'ph_total': ('pH', 'purple')
        }
        
        available_bgc = [(param, info) for param, info in bgc_params.items() 
                        if param in df.columns and df[param].notna().any()]
        
        if not available_bgc or 'depth_m' not in df.columns:
            return self._create_error_plot("BGC parameters and depth data required")
        
        # Create subplots
        n_params = len(available_bgc)
        fig = make_subplots(
            rows=1, cols=n_params,
            subplot_titles=[info[0] for _, info in available_bgc],
            shared_yaxes=True,
            horizontal_spacing=0.08
        )
        
        bgc_data = df.dropna(subset=['depth_m'])
        
        for i, (param, (label, color)) in enumerate(available_bgc, 1):
            param_data = bgc_data.dropna(subset=[param])
            
            if not param_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=param_data[param],
                        y=-param_data['depth_m'],  # Negative depth
                        mode='lines+markers',
                        name=label,
                        line=dict(color=color, width=3),
                        marker=dict(size=6, color=color)
                    ),
                    row=1, col=i
                )
        
        fig.update_yaxes(title_text="Depth (m)", row=1, col=1)
        fig.update_layout(
            title="🔬 BGC Parameters Depth Profiles",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def create_comparison_chart(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create parameter comparison chart"""
        
        # Get numeric columns for comparison
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude coordinate and id columns
        exclude_cols = ['lat', 'lon', 'depth_m', 'profile_id', 'float_id']
        compare_cols = [col for col in numeric_cols if col not in exclude_cols and df[col].notna().any()]
        
        if len(compare_cols) < 2:
            return self._create_error_plot("At least 2 parameters required for comparison")
        
        # Take first 5 parameters if too many
        compare_cols = compare_cols[:5]
        
        # Create correlation matrix
        corr_data = df[compare_cols].corr()
        
        fig = px.imshow(
            corr_data,
            text_auto=True,
            aspect="auto",
            title="📊 Parameter Correlation Matrix",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        
        fig.update_layout(
            xaxis_title="Parameters",
            yaxis_title="Parameters"
        )
        
        return fig
    
    def create_regional_summary(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create regional summary chart"""
        
        if 'region' not in df.columns:
            return self._create_error_plot("Region data required for regional summary")
        
        # Parameters to summarize
        params = ['temperature_c', 'salinity_psu', 'oxygen_umol_kg']
        available_params = [p for p in params if p in df.columns and df[p].notna().any()]
        
        if not available_params:
            return self._create_error_plot("No parameters available for regional summary")
        
        # Calculate regional statistics
        regional_stats = df.groupby('region')[available_params].agg(['mean', 'std', 'count']).round(2)
        regional_stats.columns = ['_'.join(col) for col in regional_stats.columns]
        regional_stats = regional_stats.reset_index()
        
        # Create subplots
        n_params = len(available_params)
        fig = make_subplots(
            rows=1, cols=n_params,
            subplot_titles=[p.replace('_', ' ').title() for p in available_params]
        )
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, param in enumerate(available_params, 1):
            mean_col = f"{param}_mean"
            std_col = f"{param}_std"
            
            if mean_col in regional_stats.columns:
                error_y = dict(type='data', array=regional_stats.get(std_col, [0]*len(regional_stats)))
                
                fig.add_trace(
                    go.Bar(
                        x=regional_stats['region'],
                        y=regional_stats[mean_col],
                        name=param.replace('_', ' ').title(),
                        marker_color=colors[i-1],
                        error_y=error_y
                    ),
                    row=1, col=i
                )
        
        fig.update_layout(
            title="🌍 Regional Ocean Parameter Summary",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_correlation_matrix(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create correlation matrix heatmap"""
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude non-parameter columns
        exclude_cols = ['profile_id', 'float_id', 'cycle_number', 'n_levels']
        param_cols = [col for col in numeric_cols if col not in exclude_cols and df[col].notna().sum() > 10]
        
        if len(param_cols) < 2:
            return self._create_error_plot("Insufficient numeric parameters for correlation")
        
        # Calculate correlation matrix
        corr_matrix = df[param_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="🔗 Ocean Parameter Correlations",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        
        fig.update_layout(
            xaxis_title="Ocean Parameters",
            yaxis_title="Ocean Parameters",
            height=600
        )
        
        return fig
    
    def create_folium_map(self, df: pd.DataFrame, **kwargs) -> folium.Map:
        """Create interactive Folium map for Streamlit"""
        
        if 'lat' not in df.columns or 'lon' not in df.columns:
            return folium.Map(center=[Config.MAP_CENTER_LAT, Config.MAP_CENTER_LON], zoom_start=Config.MAP_ZOOM)
        
        # Clean coordinate data
        map_data = df.dropna(subset=['lat', 'lon'])
        map_data = map_data[(map_data['lat'].between(-90, 90)) & (map_data['lon'].between(-180, 180))]
        
        if map_data.empty:
            return folium.Map(center=[Config.MAP_CENTER_LAT, Config.MAP_CENTER_LON], zoom_start=Config.MAP_ZOOM)
        
        # Calculate map center
        center_lat = map_data['lat'].mean()
        center_lon = map_data['lon'].mean()
        
        # Create map
        m = folium.Map(center=[center_lat, center_lon], zoom_start=4)
        
        # Add markers for each data point
        for _, row in map_data.iterrows():
            # Determine marker color
            color = 'blue'
            if 'status' in row:
                color = 'green' if row['status'] == 'active' else 'red'
            elif 'temperature_c' in row and pd.notna(row['temperature_c']):
                # Color by temperature
                temp = row['temperature_c']
                if temp > 25:
                    color = 'red'
                elif temp > 15:
                    color = 'orange'
                elif temp > 5:
                    color = 'blue'
                else:
                    color = 'darkblue'
            
            # Create popup content
            popup_html = self._create_popup_content(row)
            
            folium.Marker(
                [row['lat'], row['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=color, icon='info-sign'),
                tooltip=f"Float: {row.get('float_id', 'Unknown')}"
            ).add_to(m)
        
        # Add heatmap layer if temperature data available
        if 'temperature_c' in map_data.columns and map_data['temperature_c'].notna().any():
            heat_data = []
            temp_data = map_data.dropna(subset=['temperature_c'])
            
            for _, row in temp_data.iterrows():
                heat_data.append([row['lat'], row['lon'], row['temperature_c']])
            
            if heat_data:
                heat_layer = plugins.HeatMap(
                    heat_data,
                    name='Temperature Heatmap',
                    min_opacity=0.3,
                    radius=20,
                    blur=15
                )
                heat_layer.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def _create_popup_content(self, row: pd.Series) -> str:
        """Create HTML popup content for map markers"""
        
        html = f"<div style='font-family: Arial, sans-serif; font-size: 12px;'>"
        html += f"<h4 style='margin: 0; color: #2E86AB;'>🌊 ARGO Float Data</h4>"
        
        # Float information
        if 'float_id' in row:
            html += f"<p><b>Float ID:</b> {row['float_id']}</p>"
        
        # Location
        html += f"<p><b>📍 Location:</b> {row['lat']:.3f}°N, {row['lon']:.3f}°E</p>"
        
        # Date
        if 'profile_datetime' in row and pd.notna(row['profile_datetime']):
            html += f"<p><b>📅 Date:</b> {row['profile_datetime']}</p>"
        
        # Temperature
        if 'temperature_c' in row and pd.notna(row['temperature_c']):
            html += f"<p><b>🌡️ Temperature:</b> {row['temperature_c']:.1f}°C</p>"
        
        # Salinity
        if 'salinity_psu' in row and pd.notna(row['salinity_psu']):
            html += f"<p><b>🧂 Salinity:</b> {row['salinity_psu']:.1f} PSU</p>"
        
        # Depth
        if 'depth_m' in row and pd.notna(row['depth_m']):
            html += f"<p><b>🏊 Depth:</b> {row['depth_m']:.0f}m</p>"
        
        # Status
        if 'status' in row:
            status_emoji = '🟢' if row['status'] == 'active' else '🔴'
            html += f"<p><b>Status:</b> {status_emoji} {row['status'].title()}</p>"
        
        html += "</div>"
        return html
    
    def create_default_plot(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create default plot when specific type not available"""
        
        # Determine best plot type based on available data
        if 'depth_m' in df.columns and ('temperature_c' in df.columns or 'salinity_psu' in df.columns):
            return self.create_depth_profile(df, **kwargs)
        elif 'lat' in df.columns and 'lon' in df.columns:
            return self.create_trajectory_map_plotly(df, **kwargs)
        elif 'profile_datetime' in df.columns:
            return self.create_time_series(df, **kwargs)
        else:
            return self._create_empty_plot("Unable to determine appropriate visualization type")
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create empty plot with informative message"""
        
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"📊 {message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#666666"),
            align="center"
        )
        
        fig.update_layout(
            title="No Visualization Available",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400,
            paper_bgcolor='#f8f9fa'
        )
        
        return fig
    
    def _create_error_plot(self, error_message: str) -> go.Figure:
        """Create error plot with helpful message"""
        
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"❌ {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#dc3545"),
            align="center"
        )
        
        fig.update_layout(
            title="Visualization Error",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400,
            paper_bgcolor='#fff5f5'
        )
        
        return fig