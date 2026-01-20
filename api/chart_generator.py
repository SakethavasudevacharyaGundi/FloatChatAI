"""
Dynamic Chart Generation Engine for ARGO Ocean Data Explorer

This module creates interactive oceanographic visualizations based on
LLM-determined visualization strategies using Plotly and Matplotlib.

Key Features:
- Dynamic chart generation based on visualization intelligence
- Interactive Plotly charts for web deployment
- Matplotlib charts for high-quality scientific figures
- Oceanographic-specific styling and color schemes
- Integration with MCP tools for data retrieval
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import json
import numpy as np
import pandas as pd
from pathlib import Path
import base64
import io

# Visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available - install with: pip install plotly")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available - install with: pip install matplotlib seaborn")

logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    Dynamic chart generation engine for oceanographic data
    """
    
    def __init__(self):
        self.oceanographic_colorscales = self._define_oceanographic_colorscales()
        self.chart_themes = self._define_chart_themes()
        self._setup_matplotlib_style()
    
    def _define_oceanographic_colorscales(self) -> Dict[str, Any]:
        """Define color schemes suitable for oceanographic data"""
        return {
            "temperature": {
                "plotly": "RdYlBu_r",  # Red-Yellow-Blue reversed (warm to cool)
                "matplotlib": "coolwarm",
                "description": "Temperature: red (warm) to blue (cold)"
            },
            "salinity": {
                "plotly": "Viridis",
                "matplotlib": "viridis", 
                "description": "Salinity: purple (low) to yellow (high)"
            },
            "depth": {
                "plotly": "Blues_r",  # Light to dark blue (shallow to deep)
                "matplotlib": "Blues_r",
                "description": "Depth: light blue (shallow) to dark blue (deep)"
            },
            "density": {
                "plotly": "Plasma",
                "matplotlib": "plasma",
                "description": "Density: purple (low) to yellow (high)"
            },
            "oxygen": {
                "plotly": "Reds",
                "matplotlib": "Reds",
                "description": "Oxygen: white (low) to red (high)"
            },
            "default": {
                "plotly": "Viridis",
                "matplotlib": "viridis",
                "description": "Default scientific colormap"
            }
        }
    
    def _define_chart_themes(self) -> Dict[str, Any]:
        """Define chart themes for different contexts"""
        return {
            "scientific": {
                "plotly_template": "plotly_white",
                "font_family": "Arial",
                "grid_color": "lightgray",
                "background_color": "white"
            },
            "presentation": {
                "plotly_template": "presentation",
                "font_family": "Arial",
                "grid_color": "lightgray", 
                "background_color": "white"
            },
            "dark": {
                "plotly_template": "plotly_dark",
                "font_family": "Arial",
                "grid_color": "gray",
                "background_color": "rgb(17, 17, 17)"
            }
        }
    
    def _setup_matplotlib_style(self):
        """Setup matplotlib styling for scientific plots"""
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 18,
                'figure.dpi': 100
            })
    
    async def generate_chart(self, 
                           visualization_analysis: Dict[str, Any],
                           data: pd.DataFrame,
                           output_format: str = "plotly",
                           theme: str = "scientific") -> Dict[str, Any]:
        """
        Generate chart based on visualization analysis and data
        
        Args:
            visualization_analysis: Analysis from VisualizationIntelligence
            data: DataFrame with oceanographic data
            output_format: "plotly", "matplotlib", or "both"
            theme: "scientific", "presentation", or "dark"
            
        Returns:
            Dictionary with chart data, metadata, and export options
        """
        try:
            chart_type = visualization_analysis["primary_visualization"]
            viz_params = visualization_analysis["visualization_parameters"]
            
            logger.info(f"Generating {chart_type} chart with {len(data)} data points")
            
            # Prepare data for visualization
            processed_data = self._prepare_data_for_visualization(data, visualization_analysis)
            
            if not processed_data.empty:
                if output_format == "plotly" and PLOTLY_AVAILABLE:
                    chart_result = await self._generate_plotly_chart(
                        chart_type, processed_data, viz_params, theme
                    )
                elif output_format == "matplotlib" and MATPLOTLIB_AVAILABLE:
                    chart_result = await self._generate_matplotlib_chart(
                        chart_type, processed_data, viz_params, theme
                    )
                elif output_format == "both":
                    plotly_result = await self._generate_plotly_chart(
                        chart_type, processed_data, viz_params, theme
                    ) if PLOTLY_AVAILABLE else None
                    
                    matplotlib_result = await self._generate_matplotlib_chart(
                        chart_type, processed_data, viz_params, theme  
                    ) if MATPLOTLIB_AVAILABLE else None
                    
                    chart_result = {
                        "plotly": plotly_result,
                        "matplotlib": matplotlib_result,
                        "format": "both"
                    }
                else:
                    raise ValueError(f"Unsupported output format or libraries not available: {output_format}")
                
                # Add metadata
                chart_result.update({
                    "visualization_type": chart_type,
                    "data_summary": {
                        "total_points": len(processed_data),
                        "parameters": list(processed_data.columns),
                        "date_range": self._get_date_range(processed_data),
                        "depth_range": self._get_depth_range(processed_data)
                    },
                    "generation_timestamp": datetime.now().isoformat(),
                    "analysis_summary": visualization_analysis.get("reasoning", "")
                })
                
                return chart_result
            else:
                return {"error": "No data available for visualization after processing"}
                
        except Exception as e:
            logger.error(f"Chart generation failed: {str(e)}")
            return {"error": f"Chart generation failed: {str(e)}"}
    
    def _prepare_data_for_visualization(self, 
                                      data: pd.DataFrame, 
                                      analysis: Dict[str, Any]) -> pd.DataFrame:
        """Prepare and clean data for specific visualization type"""
        
        try:
            # Make a copy to avoid modifying original
            processed_data = data.copy()
            
            # Apply data processing hints from analysis
            processing_hints = analysis.get("data_processing_hints", [])
            
            if "filter_by_quality" in processing_hints:
                # Remove poor quality data if quality columns exist
                if "qc_flag" in processed_data.columns:
                    processed_data = processed_data[processed_data["qc_flag"] == 1]
            
            if "remove_outliers" in processing_hints:
                # Remove statistical outliers for numeric columns
                numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    Q1 = processed_data[col].quantile(0.25)
                    Q3 = processed_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    processed_data = processed_data[
                        (processed_data[col] >= lower_bound) & 
                        (processed_data[col] <= upper_bound)
                    ]
            
            # Handle specific chart type requirements
            chart_type = analysis["primary_visualization"]
            
            if chart_type == "temperature_profile" or chart_type == "salinity_profile":
                # Ensure depth is negative (oceanographic convention)
                if "depth_m" in processed_data.columns:
                    processed_data["depth_m"] = -abs(processed_data["depth_m"])
            
            elif chart_type == "geographic_map":
                # Ensure valid lat/lon ranges
                if "latitude" in processed_data.columns:
                    processed_data = processed_data[
                        (processed_data["latitude"] >= -90) & 
                        (processed_data["latitude"] <= 90)
                    ]
                if "longitude" in processed_data.columns:
                    processed_data = processed_data[
                        (processed_data["longitude"] >= -180) & 
                        (processed_data["longitude"] <= 180)
                    ]
            
            elif chart_type == "time_series":
                # Ensure datetime column exists and is properly formatted
                if "profile_datetime" in processed_data.columns:
                    processed_data["profile_datetime"] = pd.to_datetime(
                        processed_data["profile_datetime"]
                    )
                    processed_data = processed_data.sort_values("profile_datetime")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    async def _generate_plotly_chart(self, 
                                   chart_type: str,
                                   data: pd.DataFrame,
                                   viz_params: Dict[str, Any],
                                   theme: str) -> Dict[str, Any]:
        """Generate interactive Plotly chart"""
        
        try:
            # Set theme
            pio.templates.default = self.chart_themes[theme]["plotly_template"]
            
            if chart_type == "temperature_profile":
                fig = self._create_plotly_profile(data, "temperature_c", "depth_m", viz_params)
                
            elif chart_type == "salinity_profile":
                fig = self._create_plotly_profile(data, "salinity_psu", "depth_m", viz_params)
                
            elif chart_type == "ts_diagram":
                fig = self._create_plotly_scatter(data, "salinity_psu", "temperature_c", 
                                                "depth_m", viz_params)
                
            elif chart_type == "geographic_map":
                fig = self._create_plotly_map(data, viz_params)
                
            elif chart_type == "time_series":
                fig = self._create_plotly_timeseries(data, viz_params)
                
            elif chart_type == "depth_histogram":
                fig = self._create_plotly_histogram(data, "depth_m", viz_params)
                
            elif chart_type == "contour_plot":
                fig = self._create_plotly_contour(data, viz_params)
                
            elif chart_type == "box_plot":
                fig = self._create_plotly_boxplot(data, viz_params)
                
            else:
                # Default to temperature profile
                fig = self._create_plotly_profile(data, "temperature_c", "depth_m", viz_params)
            
            # Convert to JSON for web deployment
            chart_json = fig.to_json()
            
            # Generate static image for embedding
            try:
                img_bytes = fig.to_image(format="png", width=800, height=600)
                img_base64 = base64.b64encode(img_bytes).decode()
            except:
                img_base64 = None
                logger.warning("Could not generate static image from Plotly chart")
            
            return {
                "format": "plotly",
                "chart_json": chart_json,
                "chart_object": fig,
                "static_image_base64": img_base64,
                "interactive": True,
                "export_formats": ["html", "png", "svg", "pdf"]
            }
            
        except Exception as e:
            logger.error(f"Plotly chart generation failed: {str(e)}")
            return {"error": f"Plotly chart generation failed: {str(e)}"}
    
    def _create_plotly_profile(self, data: pd.DataFrame, x_col: str, y_col: str, 
                              viz_params: Dict[str, Any]) -> go.Figure:
        """Create temperature/salinity profile plot"""
        
        fig = go.Figure()
        
        # Group by profile if profile_id exists
        if "profile_id" in data.columns:
            for profile_id in data["profile_id"].unique()[:20]:  # Limit to 20 profiles
                profile_data = data[data["profile_id"] == profile_id]
                fig.add_trace(go.Scatter(
                    x=profile_data[x_col],
                    y=profile_data[y_col],
                    mode="lines+markers",
                    name=f"Profile {profile_id}",
                    hovertemplate=f"<b>Profile {profile_id}</b><br>" +
                                f"{x_col}: %{{x:.2f}}<br>" +
                                f"Depth: %{{y:.1f}} m<extra></extra>",
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
        else:
            # Single profile
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode="lines+markers",
                name="Profile",
                hovertemplate=f"{x_col}: %{{x:.2f}}<br>" +
                            f"Depth: %{{y:.1f}} m<extra></extra>",
                line=dict(width=3),
                marker=dict(size=6)
            ))
        
        # Update layout
        fig.update_layout(
            title=viz_params.get("title", f"{x_col.title()} Profile"),
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title="Depth (m)",
            yaxis=dict(autorange="reversed"),  # Depth increases downward
            hovermode="closest",
            template="plotly_white"
        )
        
        return fig
    
    def _create_plotly_scatter(self, data: pd.DataFrame, x_col: str, y_col: str,
                              color_col: str, viz_params: Dict[str, Any]) -> go.Figure:
        """Create scatter plot (e.g., T-S diagram)"""
        
        colorscale = self.oceanographic_colorscales.get(
            color_col.split("_")[0], self.oceanographic_colorscales["default"]
        )["plotly"]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode="markers",
            marker=dict(
                color=data[color_col],
                colorscale=colorscale,
                size=8,
                opacity=0.7,
                colorbar=dict(title=color_col.replace("_", " ").title())
            ),
            hovertemplate=f"{x_col}: %{{x:.2f}}<br>" +
                        f"{y_col}: %{{y:.2f}}<br>" +
                        f"{color_col}: %{{marker.color:.1f}}<extra></extra>",
            name="Data Points"
        ))
        
        fig.update_layout(
            title=viz_params.get("title", f"{y_col.title()} vs {x_col.title()}"),
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title=y_col.replace("_", " ").title(),
            template="plotly_white"
        )
        
        return fig
    
    def _create_plotly_map(self, data: pd.DataFrame, viz_params: Dict[str, Any]) -> go.Figure:
        """Create geographic map visualization"""
        
        # Determine color parameter
        color_col = viz_params.get("color_by", "temperature_c")
        if color_col not in data.columns:
            color_col = data.select_dtypes(include=[np.number]).columns[0]
        
        colorscale = self.oceanographic_colorscales.get(
            color_col.split("_")[0], self.oceanographic_colorscales["default"]
        )["plotly"]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scattergeo(
            lon=data["longitude"],
            lat=data["latitude"],
            mode="markers",
            marker=dict(
                color=data[color_col],
                colorscale=colorscale,
                size=8,
                opacity=0.8,
                colorbar=dict(title=color_col.replace("_", " ").title())
            ),
            hovertemplate="Lat: %{lat:.2f}<br>" +
                        "Lon: %{lon:.2f}<br>" +
                        f"{color_col}: %{{marker.color:.2f}}<extra></extra>",
            name="Float Positions"
        ))
        
        fig.update_geos(
            projection_type="natural earth",
            showland=True,
            landcolor="lightgray",
            showocean=True,
            oceancolor="lightblue"
        )
        
        fig.update_layout(
            title=viz_params.get("title", "Geographic Distribution"),
            template="plotly_white"
        )
        
        return fig
    
    def _create_plotly_timeseries(self, data: pd.DataFrame, viz_params: Dict[str, Any]) -> go.Figure:
        """Create time series visualization"""
        
        fig = go.Figure()
        
        # Get time column
        time_col = "profile_datetime"
        if time_col not in data.columns:
            time_col = data.select_dtypes(include=['datetime64']).columns[0]
        
        # Get value column
        value_col = viz_params.get("y_axis", "temperature_c")
        if value_col not in data.columns:
            value_col = data.select_dtypes(include=[np.number]).columns[0]
        
        fig.add_trace(go.Scatter(
            x=data[time_col],
            y=data[value_col],
            mode="lines+markers",
            name=value_col.replace("_", " ").title(),
            hovertemplate="Date: %{x}<br>" +
                        f"{value_col}: %{{y:.2f}}<extra></extra>",
            line=dict(width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=viz_params.get("title", f"{value_col.title()} Time Series"),
            xaxis_title="Date",
            yaxis_title=value_col.replace("_", " ").title(),
            template="plotly_white"
        )
        
        return fig
    
    def _create_plotly_histogram(self, data: pd.DataFrame, col: str, 
                                viz_params: Dict[str, Any]) -> go.Figure:
        """Create histogram visualization"""
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data[col],
            nbinsx=30,
            name="Distribution",
            hovertemplate=f"{col}: %{{x:.1f}}<br>Count: %{{y}}<extra></extra>"
        ))
        
        fig.update_layout(
            title=viz_params.get("title", f"{col.title()} Distribution"),
            xaxis_title=col.replace("_", " ").title(),
            yaxis_title="Count",
            template="plotly_white"
        )
        
        return fig
    
    def _create_plotly_contour(self, data: pd.DataFrame, viz_params: Dict[str, Any]) -> go.Figure:
        """Create contour plot visualization"""
        # This would require gridded data - simplified implementation
        fig = go.Figure()
        
        # Placeholder implementation
        fig.add_annotation(
            text="Contour plots require gridded data preprocessing",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        
        return fig
    
    def _create_plotly_boxplot(self, data: pd.DataFrame, viz_params: Dict[str, Any]) -> go.Figure:
        """Create box plot visualization"""
        
        fig = go.Figure()
        
        # Group by category if specified
        category_col = viz_params.get("x_axis", "region")
        value_col = viz_params.get("y_axis", "temperature_c")
        
        if category_col in data.columns:
            for category in data[category_col].unique():
                category_data = data[data[category_col] == category]
                fig.add_trace(go.Box(
                    y=category_data[value_col],
                    name=str(category),
                    boxpoints="outliers"
                ))
        else:
            fig.add_trace(go.Box(
                y=data[value_col],
                name=value_col.title(),
                boxpoints="outliers"
            ))
        
        fig.update_layout(
            title=viz_params.get("title", f"{value_col.title()} Distribution"),
            yaxis_title=value_col.replace("_", " ").title(),
            template="plotly_white"
        )
        
        return fig
    
    async def _generate_matplotlib_chart(self, 
                                       chart_type: str,
                                       data: pd.DataFrame,
                                       viz_params: Dict[str, Any],
                                       theme: str) -> Dict[str, Any]:
        """Generate static Matplotlib chart"""
        
        try:
            # This is a simplified implementation
            # In practice, you'd implement similar chart creation logic as Plotly
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if chart_type == "temperature_profile":
                ax.plot(data["temperature_c"], data["depth_m"], 'o-')
                ax.invert_yaxis()
                ax.set_xlabel("Temperature (°C)")
                ax.set_ylabel("Depth (m)")
            
            elif chart_type == "geographic_map":
                scatter = ax.scatter(data["longitude"], data["latitude"], 
                                   c=data.get("temperature_c", 'blue'))
                plt.colorbar(scatter, ax=ax, label="Temperature (°C)")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
            
            else:
                # Default scatter plot
                ax.scatter(data.iloc[:, 0], data.iloc[:, 1])
                ax.set_xlabel(data.columns[0])
                ax.set_ylabel(data.columns[1])
            
            ax.set_title(viz_params.get("title", "Oceanographic Data"))
            plt.tight_layout()
            
            # Convert to base64 for embedding
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return {
                "format": "matplotlib",
                "static_image_base64": img_base64,
                "interactive": False,
                "export_formats": ["png", "svg", "pdf"]
            }
            
        except Exception as e:
            logger.error(f"Matplotlib chart generation failed: {str(e)}")
            return {"error": f"Matplotlib chart generation failed: {str(e)}"}
    
    def _get_date_range(self, data: pd.DataFrame) -> Dict[str, str]:
        """Get date range from data"""
        date_cols = data.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            return {
                "start": data[date_col].min().isoformat(),
                "end": data[date_col].max().isoformat()
            }
        return {}
    
    def _get_depth_range(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get depth range from data"""
        if "depth_m" in data.columns:
            return {
                "min_depth": float(data["depth_m"].min()),
                "max_depth": float(data["depth_m"].max())
            }
        return {}

# Global instance
chart_generator = ChartGenerator()