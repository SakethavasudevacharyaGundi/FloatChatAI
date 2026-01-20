import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import asyncio
from datetime import datetime
import sys
import os

# Add project path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gemini_orchestrator import GeminiOrchestrator
except ImportError as e:
    st.error(f"Failed to import GeminiOrchestrator: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="FloatChat AI Dashboard",
    page_icon="🌊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
    }
    
    .nav-button {
        background: #2a5298;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0 0.25rem;
        cursor: pointer;
    }
    
    .floating-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
    }
    
    .chat-message {
        margin: 0.5rem 0;
        padding: 0.75rem;
        border-radius: 10px;
    }
    
    .chat-user {
        background: rgba(100, 149, 237, 0.2);
        border-left: 3px solid #6495ED;
    }
    
    .chat-ai {
        background: rgba(50, 205, 50, 0.2);
        border-left: 3px solid #32CD32;
    }
    
    .utility-icon {
        font-size: 1.2rem;
        opacity: 0.7;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .css-1y4p8pa {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Custom heading styles */
    .section-heading {
        background: #1a1a1a;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 8px;
        margin: 16px 0 8px 0;
        font-weight: 600;
        font-size: 1.1rem;
        border-left: 4px solid #2a5298;
    }
    
    .subsection-heading {
        background: #2a2a2a;
        color: #e0e0e0;
        padding: 6px 12px;
        border-radius: 6px;
        margin: 12px 0 6px 0;
        font-weight: 500;
        font-size: 1rem;
        border-left: 3px solid #4a69bd;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load real ARGO float data from CSV files"""
    try:
        # Try to load real ARGO data
        import os
        
        # Look for ARGO profiles data
        argo_profiles_path = "data/argo_profiles.csv"
        argo_floats_path = "data/argo_floats.csv"
        
        if os.path.exists(argo_profiles_path):
            # Load the data silently without showing loading messages
            data = pd.read_csv(argo_profiles_path)
            
            # Clean and prepare data
            data = data.dropna(subset=['latitude', 'longitude'])
            
            # Rename columns to match expected format
            column_mapping = {
                'profile_id': 'profile_index',
                'timestamp': 'date'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns:
                    data = data.rename(columns={old_col: new_col})
            
            # Ensure we have a profile_index column
            if 'profile_index' not in data.columns and 'profile_id' in data.columns:
                data['profile_index'] = data['profile_id']
            elif 'profile_index' not in data.columns:
                data['profile_index'] = data.index
            
            # Convert date column if exists
            if 'date' in data.columns:
                try:
                    data['date'] = pd.to_datetime(data['date'])
                except:
                    pass
            
            # Limit data for performance and return quietly
            return data.head(5000)  # Limit for performance
            
        elif os.path.exists(argo_floats_path):
            # Load ARGO floats data silently
            data = pd.read_csv(argo_floats_path)
            return data.head(5000)
            
        else:
            # Generate sample data if no real data available (silently)
            import numpy as np
            np.random.seed(42)
            
            n_points = 500
            profiles = np.random.randint(1, 50, n_points)
            
            # Realistic ocean coordinates
            latitudes = np.random.uniform(-60, 60, n_points)
            longitudes = np.random.uniform(-180, 180, n_points)
            
            # Depth-dependent temperature and salinity
            depths = np.random.uniform(0, 2000, n_points)
            temperatures = 25 - (depths / 100) + np.random.normal(0, 2, n_points)
            salinities = 35 + np.random.normal(0, 0.5, n_points)
            pressures = depths * 10 + np.random.normal(0, 5, n_points)
            
            # Dates
            dates = pd.date_range('2023-01-01', periods=n_points, freq='D')
            
            return pd.DataFrame({
                'profile_index': profiles,
                'float_id': profiles,  # Same as profile for simplicity
                'latitude': latitudes,
                'longitude': longitudes,
                'temperature': temperatures,
                'salinity': salinities,
                'pressure': pressures,
                'depth': depths,
                'date': dates
            })
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def initialize_orchestrator():
    """Initialize the Gemini orchestrator"""
    try:
        if 'orchestrator' not in st.session_state:
            orchestrator = GeminiOrchestrator()
            if orchestrator:
                st.session_state.orchestrator = orchestrator
                return True
        return True
    except Exception as e:
        st.error(f"Failed to initialize orchestrator: {e}")
        return False

def create_satellite_map_with_floats(data):
    """Create a map of ARGO float distribution using the provided fixed coordinates."""
    try:
        # Fixed coordinate pairs (latitude, longitude)
        coords = [
            (-12.368, 47.239), (-12.335, 47.467), (-12.162, 47.161), (-12.083, 47.147),
            (-11.998, 47.161), (-11.861, 46.745), (-11.834, 46.818), (-11.782, 47.207),
            (-11.775, 47.377), (-11.759, 46.567), (-11.756, 47.298), (-11.729, 46.642),
            (-11.705, 47.382), (-11.565, 47.577), (-11.534, 47.242), (-11.469, 48.363),
            (-11.456, 47.485), (-11.309, 47.268), (-11.298, 48.441), (-11.295, 44.251),
            (-11.270, 48.626), (-11.254, 47.419), (-11.151, 41.411), (-11.113, 44.803),
            (-11.049, 46.409), (-11.047, 45.519), (-11.039, 43.669), (-10.958, 41.303),
            (-10.918, 47.340), (-10.838, 44.305), (-10.768, 48.779), (-10.658, 42.343),
            (-10.644, 41.592), (-10.635, 44.602), (-10.621, 42.856), (-10.535, 44.277),
            (-10.500, 48.909), (-10.480, 54.816), (-10.462, 54.622), (-10.396, 41.922),
            (-10.326, 54.926), (-10.326, 42.032), (-10.321, 49.139), (-10.318, 54.519),
            (-10.286, 54.846), (-10.281, 41.028), (-10.250, 54.724), (-10.232, 54.476),
            (-10.224, 54.390), (-10.221, 54.957), (-10.169, 40.855), (-10.158, 54.561),
            (-10.151, 54.768), (-10.145, 54.781), (-10.144, 40.745), (-10.114, 54.538),
            (-10.105, 55.066), (-10.100, 40.879), (-10.093, 40.764), (-10.059, 55.142),
            (-10.057, 40.944), (-10.054, 54.320), (-10.024, 55.666), (-9.989, 55.202),
            (-9.987, 40.709), (-9.986, 55.492), (-9.976, 49.439), (-9.925, 55.695),
            (-9.918, 40.713), (-9.912, 54.050), (-9.875, 40.533), (-9.857, 55.953),
            (-9.825, 55.439), (-9.767, 50.009), (-9.746, 50.338), (-9.743, 55.198),
            (-9.741, 40.353), (-9.712, 55.321), (-9.687, 49.661), (-9.638, 53.872),
            (-9.605, 40.155), (-9.554, 53.351), (-9.424, 51.921), (-9.414, 40.070),
            (-9.383, 52.278), (-9.374, 40.057), (-9.357, 52.882), (-9.352, 53.179),
            (-9.329, 51.539), (-9.278, 51.532), (-9.265, 51.970), (-9.211, 51.855),
            (-9.185, 51.276), (-9.121, 50.968), (-9.096, 51.515), (-9.089, 39.889),
            (-8.747, 39.816), (-8.460, 39.717), (-8.046, 39.820)
        ]

        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(size=6, color='#FF6B6B', opacity=0.85),
            text=[f"Lat: {lat:.3f}, Lon: {lon:.3f}" for lat, lon in zip(lats, lons)],
            hovertemplate='%{text}<extra></extra>',
            name='ARGO Floats'
        ))

        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        fig.update_layout(
            geo=dict(
                projection_type='natural earth',
                showcountries=True,
                showsubunits=True,
                lataxis=dict(range=[min(lats)-2, max(lats)+2]),
                lonaxis=dict(range=[min(lons)-2, max(lons)+2]),
                center=dict(lat=center_lat, lon=center_lon)
            ),
            margin=dict(r=0, t=0, l=0, b=0),
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    except Exception as e:
        st.error(f"Error creating satellite map: {e}")
        return create_fallback_map()

def create_fallback_map():
    """Create a simple fallback map when satellite map fails"""
    fig = go.Figure()
    
    # Add a basic world map outline
    fig.add_trace(go.Scattergeo(
        lon=[-180, 180, 180, -180, -180],
        lat=[-90, -90, 90, 90, -90],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.3)'),
        showlegend=False
    ))
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular',
            bgcolor='rgba(0,0,0,0.8)'
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_map(data):
    """Create ARGO Float Map"""
    return create_satellite_map_with_floats(data)

def create_depth_profile(data, float_id=None):
    """Create depth analysis chart matching the described oceanographic shape."""
    if data.empty:
        return create_empty_chart("No data available")

    if 'temperature' not in data.columns and 'temp' not in ''.join(data.columns).lower():
        return create_empty_chart("Missing temperature column")

    # Choose depth column; convert pressure≈depth if needed
    if 'depth' in data.columns:
        depth_series = data['depth'].copy()
        depth_label = 'Depth (m)'
    elif 'pressure' in data.columns:
        depth_series = data['pressure'].copy()  # ~ dbar ≈ m
        depth_label = 'Depth (m)'
    else:
        return create_empty_chart("Missing depth/pressure column")

    # Select subset
    df = data.copy()
    if float_id and 'float_id' in df.columns:
        df = df[df['float_id'] == float_id]
    if df.empty:
        df = data.sample(min(500, len(data)))

    # Normalize columns
    temp_col = 'temperature' if 'temperature' in df.columns else [c for c in df.columns if 'temp' in c.lower()][0]
    dcol = 'depth' if 'depth' in df.columns else 'pressure'
    df = df[[temp_col, dcol]].dropna().rename(columns={temp_col: 'temperature', dcol: 'depth_m'})
    if dcol == 'pressure':
        df['depth_m'] = df['depth_m']  # ~1 dbar ≈ 1 m

    # Sort by depth increasing
    df = df.sort_values('depth_m')

    # Smooth to reveal thermocline
    window = max(5, min(51, len(df)//20*2+1))
    df['temp_smooth'] = df['temperature'].rolling(window, center=True, min_periods=1).mean()

    # Gradient for thermocline detection
    df['dT'] = df['temp_smooth'].diff()
    df['dZ'] = df['depth_m'].diff().replace(0, 1e-6)
    df['gradient'] = (df['dT'] / df['dZ']).abs()
    if not df['gradient'].dropna().empty:
        therm_idx = int(df['gradient'].idxmax())
        therm_depth = float(df.loc[therm_idx, 'depth_m'])
        band_top = max(0.0, therm_depth - 75.0)
        band_bot = therm_depth + 75.0
    else:
        band_top, band_bot = 100.0, 250.0

    # Build figure
    fig = go.Figure()
    # Scatter points (noisy surface)
    fig.add_trace(go.Scatter(
        x=df['temperature'], y=df['depth_m'], mode='markers',
        marker=dict(size=3, color='rgba(255,255,255,0.25)'), name='Samples', showlegend=False
    ))
    # Smoothed curve
    fig.add_trace(go.Scatter(
        x=df['temp_smooth'], y=df['depth_m'], mode='lines',
        line=dict(color='#FF6B6B', width=3), name='Temperature (smoothed)'
    ))
    # Surface mixed layer band (0–50 m)
    fig.add_shape(type='rect', xref='paper', yref='y', x0=0, x1=1, y0=0, y1=50,
                  fillcolor='rgba(100,149,237,0.15)', line=dict(width=0), layer='below')
    # Thermocline band
    fig.add_shape(type='rect', xref='paper', yref='y', x0=0, x1=1, y0=band_top, y1=band_bot,
                  fillcolor='rgba(255,165,0,0.12)', line=dict(width=0), layer='below')
    # Deep stable layer hint
    fig.add_shape(type='rect', xref='paper', yref='y', x0=0, x1=1, y0=band_bot, y1=max(df['depth_m'].max(), band_bot+200),
                  fillcolor='rgba(144,238,144,0.08)', line=dict(width=0), layer='below')

    fig.update_layout(
        title="Temperature vs Depth (Surface mixed layer → Thermocline → Deep stable)",
        xaxis_title="Temperature (°C)", yaxis_title=depth_label,
        yaxis=dict(autorange='reversed'), template='plotly_dark', height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )

    return fig

def create_salinity_profile(data, float_id=None):
    """Create salinity profile matching the halocline description."""
    if data.empty:
        return create_empty_chart("No data available")

    if 'salinity' not in data.columns and not any('sal' in c.lower() for c in data.columns):
        return create_empty_chart("Missing salinity column")

    # Choose depth axis
    if 'depth' in data.columns:
        dcol = 'depth'; depth_label = 'Depth (m)'
    elif 'pressure' in data.columns:
        dcol = 'pressure'; depth_label = 'Depth (m)'
    else:
        return create_empty_chart("Missing depth/pressure column")

    df = data.copy()
    if float_id and 'float_id' in df.columns:
        df = df[df['float_id'] == float_id]
    if df.empty:
        df = data.sample(min(500, len(data)))

    sal_col = 'salinity' if 'salinity' in df.columns else [c for c in df.columns if 'sal' in c.lower()][0]
    df = df[[sal_col, dcol]].dropna().rename(columns={sal_col: 'salinity', dcol: 'depth_m'})
    if dcol == 'pressure':
        df['depth_m'] = df['depth_m']

    df = df.sort_values('depth_m')
    window = max(5, min(51, len(df)//20*2+1))
    df['sal_smooth'] = df['salinity'].rolling(window, center=True, min_periods=1).mean()

    # Halocline detection via gradient
    df['dS'] = df['sal_smooth'].diff(); df['dZ'] = df['depth_m'].diff().replace(0, 1e-6)
    df['gradient'] = (df['dS'] / df['dZ']).abs()
    if not df['gradient'].dropna().empty:
        halo_idx = int(df['gradient'].idxmax())
        halo_depth = float(df.loc[halo_idx, 'depth_m'])
        htop, hbot = max(0.0, halo_depth - 50.0), halo_depth + 50.0
    else:
        htop, hbot = 75.0, 175.0

    fig = go.Figure()
    # Noisy surface scatter
    fig.add_trace(go.Scatter(
        x=df['salinity'], y=df['depth_m'], mode='markers',
        marker=dict(size=3, color='rgba(255,255,255,0.25)'), name='Samples', showlegend=False
    ))
    # Smoothed halocline curve
    fig.add_trace(go.Scatter(
        x=df['sal_smooth'], y=df['depth_m'], mode='lines',
        line=dict(color='#00BFA6', width=3), name='Salinity (smoothed)'
    ))
    # Surface mixed layer band (flat/noisy)
    fig.add_shape(type='rect', xref='paper', yref='y', x0=0, x1=1, y0=0, y1=40,
                  fillcolor='rgba(100,149,237,0.15)', line=dict(width=0), layer='below')
    # Halocline band
    fig.add_shape(type='rect', xref='paper', yref='y', x0=0, x1=1, y0=htop, y1=hbot,
                  fillcolor='rgba(255,165,0,0.12)', line=dict(width=0), layer='below')
    # Deep stable
    fig.add_shape(type='rect', xref='paper', yref='y', x0=0, x1=1, y0=hbot, y1=max(df['depth_m'].max(), hbot+200),
                  fillcolor='rgba(144,238,144,0.08)', line=dict(width=0), layer='below')

    fig.update_layout(
        title="Salinity vs Depth (Surface → Halocline → Deep stable)",
        xaxis_title="Salinity (PSU)", yaxis_title=depth_label,
        yaxis=dict(autorange='reversed'), template='plotly_dark', height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )

    return fig

def create_ts_diagram(data, float_id=None):
    """Create Temperature-Salinity diagram"""
    if float_id:
        float_data = data[data['float_id'] == float_id]
    else:
        float_data = data.sample(min(1000, len(data)))  # Sample more points for T-S diagram
    
    # Check if we have the required columns
    if 'temperature' not in float_data.columns or 'salinity' not in float_data.columns:
        # Try alternative column names
        temp_col = None
        sal_col = None
        for col in float_data.columns:
            if 'temp' in col.lower():
                temp_col = col
            elif 'sal' in col.lower():
                sal_col = col
        
        if temp_col is None or sal_col is None:
            return None
    else:
        temp_col = 'temperature'
        sal_col = 'salinity'
    
    # Clean data - remove NaN values
    clean_data = float_data.dropna(subset=[temp_col, sal_col])
    
    if clean_data.empty:
        return None
    
    fig = go.Figure()
    
    # Color by depth if available
    if 'depth' in clean_data.columns:
        fig.add_trace(go.Scatter(
            x=clean_data[sal_col],
            y=clean_data[temp_col],
            mode='markers',
            marker=dict(
                size=6,
                color=clean_data['depth'],
                colorscale='Viridis',
                colorbar=dict(title="Depth (m)"),
                opacity=0.7
            ),
            text=[f"Depth: {d:.1f}m<br>Temp: {t:.2f}°C<br>Salinity: {s:.2f}" 
                  for d, t, s in zip(clean_data['depth'], clean_data[temp_col], clean_data[sal_col])],
            hovertemplate='%{text}<extra></extra>',
            name='T-S Points'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=clean_data[sal_col],
            y=clean_data[temp_col],
            mode='markers',
            marker=dict(
                size=6,
                color='#FF6B6B',
                opacity=0.7
            ),
            text=[f"Temp: {t:.2f}°C<br>Salinity: {s:.2f}" 
                  for t, s in zip(clean_data[temp_col], clean_data[sal_col])],
            hovertemplate='%{text}<extra></extra>',
            name='T-S Points'
        ))
    
    fig.update_layout(
        title="Temperature-Salinity Diagram",
        xaxis_title="Salinity (PSU)",
        yaxis_title="Temperature (°C)",
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig

def show_dashboard_page(data):
    """Display the main dashboard page"""
    # Remove the Quick Chat card - full width map and visualizations
    
    # ARGO Float Distribution Map Section
    st.markdown('<div class="section-heading">ARGO Float Distribution Map</div>', unsafe_allow_html=True)
    
    # Display current chart or default satellite map
    if 'current_chart' in st.session_state:
        st.plotly_chart(st.session_state.current_chart, use_container_width=True, key="sidebar_current_chart")
    else:
        # Default satellite map
        map_fig = create_satellite_map_with_floats(data)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True, key="sidebar_satellite_map")
        else:
            st.info("Loading map view...")
    
    # Generated Visualizations Section
    if hasattr(st.session_state, 'query_results') and st.session_state.query_results:
        if 'task_results' in st.session_state.query_results:
            task_results = st.session_state.query_results['task_results']
            
            if 'visualizations' in task_results and task_results['visualizations']:
                st.markdown('<div class="section-heading">Generated Visualizations</div>', unsafe_allow_html=True)
                
                for viz_name, viz_data in task_results['visualizations'].items():
                    if isinstance(viz_data, dict) and 'figure' in viz_data:
                        st.markdown(f"#### {viz_data.get('description', viz_name)}")
                        try:
                            import plotly.io
                            fig = plotly.io.from_json(viz_data['figure'])
                            st.plotly_chart(fig, use_container_width=True, key=f"sidebar_viz_{viz_name}")
                        except Exception as e:
                            st.error(f"Error displaying {viz_name}: {e}")
                    elif isinstance(viz_data, dict) and viz_data.get('insufficient_data'):
                        st.warning(f"**{viz_name}**: {viz_data.get('reason', 'Insufficient data')}")
                    elif isinstance(viz_data, dict) and 'error' in viz_data:
                        st.error(f"**{viz_name}**: {viz_data['error']}")
    
    # Handle quick queries from sidebar
    if hasattr(st.session_state, 'quick_query'):
        user_input = st.session_state.quick_query
        delattr(st.session_state, 'quick_query')
        
        if user_input and st.session_state.orchestrator:
            with st.spinner("Analyzing..."):
                try:
                    # Process query with orchestrator
                    result = asyncio.run(st.session_state.orchestrator.process_query(user_input))
                    
                    # Display response
                    response = result.get('response', 'No response generated')
                    st.success(response)
                    
                    # Store results for visualization
                    st.session_state.query_results = result
                    
                except Exception as e:
                    st.error(f"Query processing failed: {e}")

def show_chatbot_page():
    """Display the dedicated chatbot page with ChatGPT-style interface"""
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chatbot header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h2>🤖 FloatChat AI Assistant</h2>
        <p>Ask questions about ocean data, request analyses, or explore ARGO float insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container with scrollable area
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if message['type'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
    
    # Fixed chat input at bottom
    user_input = st.chat_input("Ask about ocean data, request analysis, or explore insights...")
    
    # Process user input
    if user_input and st.session_state.orchestrator:
        # Add user message to chat history
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Show user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process and show AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Process query with orchestrator
                    result = asyncio.run(st.session_state.orchestrator.process_query(user_input))
                    
                    # Display response
                    response = result.get('response', 'No response generated')
                    st.write(response)
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'content': response,
                        'timestamp': datetime.now()
                    })
                    
                    # Store results for potential visualization use
                    st.session_state.query_results = result
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'content': error_msg,
                        'timestamp': datetime.now()
                    })
        
        # Auto-scroll to bottom by rerunning
        st.rerun()

def main():
    """Main application"""
    
    # Initialize orchestrator
    if not initialize_orchestrator():
        st.error("Failed to initialize system. Please refresh the page.")
        return
    
    # Load data
    data = load_sample_data()
    
    # Initialize page state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'dashboard'

    # Main Header - moved to top
    st.markdown("""
    <div class="main-header">
        <h1>🌊 FloatChat AI</h1>
        <p style="margin: 0; font-size: 1.2rem; opacity: 0.9;">Ocean Data Assistant - Intelligent Analysis of ARGO Float Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Top navigation bar
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)  # Simple spacer

    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="section-heading">Navigation</div>', unsafe_allow_html=True)
        
        # Page selector
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏠 Dashboard", use_container_width=True):
                st.session_state.current_page = 'dashboard'
                st.rerun()
                
        with col2:
            if st.button("💬 Chatbot", use_container_width=True):
                st.session_state.current_page = 'chatbot'
                st.rerun()
        
        st.markdown("---")
        
        if st.session_state.current_page == 'dashboard':
            st.markdown('<div class="section-heading">Interactive Charts</div>', unsafe_allow_html=True)
            
            # Chart type selector
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Temperature Map", "Salinity Profile", "Depth Analysis", "T-S Diagram"]
            )
            
            if st.button("📊 Generate Chart"):
                if chart_type == "Temperature Map":
                    fig = create_satellite_map_with_floats(data)
                    if fig:
                        st.session_state.current_chart = fig
                elif chart_type == "Salinity Profile":
                    fig = create_salinity_profile(data)
                    if fig:
                        st.session_state.current_chart = fig
                elif chart_type == "Depth Analysis":
                    fig = create_depth_profile(data)
                    if fig:
                        st.session_state.current_chart = fig
                elif chart_type == "T-S Diagram":
                    fig = create_ts_diagram(data)
                    if fig:
                        st.session_state.current_chart = fig
                    else:
                        st.error("Unable to generate T-S diagram - missing temperature or salinity data")
                else:
                    st.info(f"Generating {chart_type}...")
            
            st.markdown("---")
            
            st.markdown('<div class="section-heading">Quick Chat Actions</div>', unsafe_allow_html=True)
            
            # Quick action buttons
            st.markdown("**Quick Actions:**")
            if st.button("📈 Calculate Statistics"):
                st.session_state.quick_query = "calculate statistics"
                
            if st.button("🌡️ Highest Temperature"):
                st.session_state.quick_query = "highest temperature"
                
            if st.button("🧂 Max Salinity"):
                st.session_state.quick_query = "maximum salinity"
            
        if st.button("📍 Geographic Analysis"):
            st.session_state.quick_query = "analyze geographic distribution"
        
        st.markdown("---")
        
        # Data context
        st.markdown('<div class="section-heading">Data Overview & Filters</div>', unsafe_allow_html=True)
        
        if not data.empty:
            # Real data metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Profiles", len(data))
                if 'float_id' in data.columns:
                    st.metric("Unique Floats", data['float_id'].nunique())
                elif 'profile_index' in data.columns:
                    st.metric("Profile Groups", data['profile_index'].nunique())
            
            with col2:
                if 'region' in data.columns:
                    st.metric("Ocean Regions", data['region'].nunique())
                if 'date' in data.columns or 'timestamp' in data.columns:
                    date_col = 'date' if 'date' in data.columns else 'timestamp'
                    try:
                        date_range = pd.to_datetime(data[date_col])
                        st.metric("Data Range", f"{date_range.min().strftime('%Y-%m')} to {date_range.max().strftime('%Y-%m')}")
                    except:
                        st.metric("Data Range", "Available")
            
            # Geographic filter based on real data
            if 'latitude' in data.columns and 'longitude' in data.columns:
                st.markdown('<div class="subsection-heading">Geographic Bounds</div>', unsafe_allow_html=True)
                
                lat_min, lat_max = data['latitude'].min(), data['latitude'].max()
                lon_min, lon_max = data['longitude'].min(), data['longitude'].max()
                
                lat_range = st.slider(
                    "Latitude Range", 
                    float(lat_min), float(lat_max), 
                    (float(lat_min), float(lat_max)),
                    step=0.1
                )
                
                lon_range = st.slider(
                    "Longitude Range", 
                    float(lon_min), float(lon_max), 
                    (float(lon_min), float(lon_max)),
                    step=0.1
                )
            
            # Ocean region filter if available
            if 'region' in data.columns:
                st.markdown('<div class="subsection-heading">Ocean Region</div>', unsafe_allow_html=True)
                regions = ['All'] + sorted(data['region'].dropna().unique().tolist())
                selected_region = st.selectbox("Filter by Region", regions)
            
            # Depth range filter
            if 'depth' in data.columns:
                st.markdown('<div class="subsection-heading">Depth Range</div>', unsafe_allow_html=True)
                depth_min, depth_max = data['depth'].min(), data['depth'].max()
                depth_range = st.slider(
                    "Depth (m)", 
                    float(depth_min), float(depth_max), 
                    (float(depth_min), min(2000.0, float(depth_max))),
                    step=10.0
                )
            
            # Data quality filter if available
            if 'quality_flag' in data.columns:
                st.markdown('<div class="subsection-heading">Data Quality</div>', unsafe_allow_html=True)
                quality_options = ['All'] + sorted(data['quality_flag'].dropna().unique().tolist())
                quality_filter = st.selectbox("Quality Flag", quality_options)

    # Main content area - page routing
    if st.session_state.current_page == 'dashboard':
        show_dashboard_page(data)
    elif st.session_state.current_page == 'chatbot':
        show_chatbot_page()

if __name__ == "__main__":
    main()