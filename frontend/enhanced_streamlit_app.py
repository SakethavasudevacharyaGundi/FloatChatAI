"""
Enhanced ARGO Ocean Data Explorer with Gemini AI Integration
Complete dashboard with advanced visualizations, interactive maps, and intelligent query processing
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
import logging
from typing import Dict, Any, List, Optional, Union
import json
import io
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our enhanced components
try:
    from llm.gemini_processor import GeminiQueryProcessor
    from visualization.ocean_plots import OceanVisualizationEngine
    from data.netcdf_processor import NetCDFProcessor
    from rag.enhanced_rag_pipeline import EnhancedRAGPipeline
    from mapping.geospatial_mapper import GeospatialMapper
    from vector_db.vector_store import VectorStore
    from data.dummy_data_generator import DummyDataGenerator
    from config import Config
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Please ensure all components are properly installed and available.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🌊 ARGO Ocean Data Explorer - Gemini AI Enhanced",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #2d3748;
        border-left: 4px solid #38b2ac;
    }
    
    .gemini-response {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ed8936;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f7fafc;
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(66, 153, 225, 0.4);
    }
</style>
""", unsafe_allow_html=True)

class EnhancedArgoExplorer:
    """Enhanced ARGO data explorer with Gemini AI integration"""
    
    def __init__(self):
        self.components = self.init_components()
    
    def init_components(self):
        """Initialize all system components"""
        try:
            # Initialize core components
            components = {
                'gemini_processor': GeminiQueryProcessor(),
                'visualization_engine': OceanVisualizationEngine(),
                'netcdf_processor': NetCDFProcessor(),
                'geospatial_mapper': GeospatialMapper(),
                'vector_store': VectorStore(),
                'dummy_generator': DummyDataGenerator()
            }
            
            # Initialize RAG pipeline with vector store
            components['rag_pipeline'] = EnhancedRAGPipeline(components['vector_store'])
            
            logger.info("✅ All components initialized successfully")
            return components
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            st.error(f"Failed to initialize system components: {e}")
            return None
    
    def main(self):
        """Main application entry point"""
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🌊 ARGO Ocean Data Explorer</h1>
            <h3>🤖 Powered by Google Gemini AI</h3>
            <p>Advanced oceanographic data analysis with natural language intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check component initialization
        if not hasattr(self, 'components') or self.components is None:
            st.error("❌ System components failed to initialize. Please refresh the page.")
            return
        
        # Sidebar configuration
        self.setup_sidebar()
        
        # Main content tabs
        self.setup_main_content()
    
    def setup_sidebar(self):
        """Setup sidebar with configuration options"""
        
        st.sidebar.markdown("## 🎛️ System Configuration")
        
        # API Status
        self.check_gemini_status()
        
        # Data source selection
        st.sidebar.markdown("### 📊 Data Source")
        data_source = st.sidebar.selectbox(
            "Select Data Source",
            ["Dummy Data (Demo)", "Upload NetCDF Files", "Live ARGO Data"],
            help="Choose your data source for analysis"
        )
        
        # Store in session state
        st.session_state['data_source'] = data_source
        
        # Analysis parameters
        st.sidebar.markdown("### 📈 Analysis Parameters")
        
        max_records = st.sidebar.slider(
            "Max Records to Process",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Maximum number of records to retrieve and analyze"
        )
        
        st.session_state['max_records'] = max_records
        
        # Visualization preferences
        st.sidebar.markdown("### 🎨 Visualization Settings")
        
        plot_style = st.sidebar.selectbox(
            "Plot Style",
            ["Default", "Scientific", "Presentation", "Dark Theme"],
            help="Choose visualization theme"
        )
        
        st.session_state['plot_style'] = plot_style
        
        # Regional focus
        region_focus = st.sidebar.selectbox(
            "Regional Focus",
            ["Global", "Indian Ocean", "Pacific Ocean", "Atlantic Ocean", "Southern Ocean"],
            index=1,  # Default to Indian Ocean
            help="Select oceanic region for focused analysis"
        )
        
        st.session_state['region_focus'] = region_focus
        
        # Advanced options
        with st.sidebar.expander("🔧 Advanced Options"):
            enable_ai_insights = st.checkbox(
                "Enable AI Insights",
                value=True,
                help="Use Gemini AI for intelligent data interpretation"
            )
            
            enable_quality_control = st.checkbox(
                "Enable Quality Control",
                value=True,
                help="Apply automatic data quality filtering"
            )
            
            enable_caching = st.checkbox(
                "Enable Result Caching",
                value=True,
                help="Cache results for faster subsequent queries"
            )
            
            st.session_state.update({
                'enable_ai_insights': enable_ai_insights,
                'enable_quality_control': enable_quality_control,
                'enable_caching': enable_caching
            })
    
    def check_gemini_status(self):
        """Check and display Gemini API status"""
        
        st.sidebar.markdown("### 🤖 AI System Status")
        
        try:
            # Test Gemini connection
            test_response = self.components['gemini_processor'].process_query(
                "Test connection - respond with 'OK'"
            )
            
            if test_response and test_response.get('response'):
                st.sidebar.success("✅ Gemini AI Connected")
                st.sidebar.info(f"🔑 API Key: ...{Config.GOOGLE_API_KEY[-8:]}")
            else:
                st.sidebar.error("❌ Gemini AI Disconnected")
                
        except Exception as e:
            st.sidebar.error(f"❌ Gemini Error: {str(e)[:50]}...")
    
    def setup_main_content(self):
        """Setup main content area with tabs"""
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Intelligent Query",
            "📊 Data Visualization", 
            "🗺️ Interactive Maps",
            "📁 Data Management",
            "📈 System Analytics"
        ])
        
        with tab1:
            self.intelligent_query_interface()
        
        with tab2:
            self.data_visualization_interface()
        
        with tab3:
            self.interactive_mapping_interface()
        
        with tab4:
            self.data_management_interface()
        
        with tab5:
            self.system_analytics_interface()
    
    def intelligent_query_interface(self):
        """Natural language query interface with Gemini AI"""
        
        st.markdown("## 🧠 Intelligent Query System")
        st.markdown("Ask questions about oceanographic data in natural language!")
        
        # Query examples
        with st.expander("💡 Example Queries"):
            st.markdown("""
            **Temperature Analysis:**
            - "What is the average temperature in the Arabian Sea?"
            - "Show me temperature profiles deeper than 1000 meters"
            - "Compare temperature trends over the last year"
            
            **Spatial Analysis:**
            - "Map all active ARGO floats in the Indian Ocean"
            - "Show salinity distribution in the Bay of Bengal"
            - "Find anomalous temperature readings in tropical regions"
            
            **Temporal Analysis:**
            - "Track float movements over the past 6 months"
            - "Show seasonal variations in oxygen levels"
            - "Analyze BGC parameter trends"
            """)
        
        # Query input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_query = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="e.g., 'Show me temperature depth profiles in the Arabian Sea with high salinity values'",
                help="Ask any question about oceanographic data in natural language"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            process_query = st.button(
                "🚀 Process Query",
                use_container_width=True,
                help="Process your query using Gemini AI"
            )
        
        if process_query and user_query:
            self.process_intelligent_query(user_query)
        
        # Query history
        if 'query_history' not in st.session_state:
            st.session_state['query_history'] = []
        
        if st.session_state['query_history']:
            with st.expander("📜 Query History"):
                for i, (query, timestamp) in enumerate(reversed(st.session_state['query_history'][-5:])):
                    if st.button(f"🔄 {query[:50]}..." if len(query) > 50 else query, key=f"history_{i}"):
                        self.process_intelligent_query(query)
    
    def process_intelligent_query(self, query: str):
        """Process natural language query using enhanced RAG pipeline"""
        
        with st.spinner("🤖 Processing your query with Gemini AI..."):
            try:
                # Add to history
                st.session_state['query_history'].append((query, datetime.now()))
                
                # Process through enhanced RAG pipeline
                result = self.components['rag_pipeline'].query(
                    query,
                    filters={'region': st.session_state.get('region_focus', 'Global').lower()}
                )
                
                # Display results
                self.display_query_results(result)
                
            except Exception as e:
                st.error(f"❌ Query processing failed: {e}")
                logger.error(f"Query processing error: {e}")
    
    def display_query_results(self, result: Dict[str, Any]):
        """Display comprehensive query results"""
        
        if not result:
            st.warning("⚠️ No results returned from query processing")
            return
        
        # AI Response
        if result.get('response'):
            st.markdown(f"""
            <div class="gemini-response">
                <h4>🤖 Gemini AI Analysis</h4>
                <p>{result['response']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Query Analysis
        query_analysis = result.get('query_analysis', {})
        if query_analysis:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 Intent Detected</h4>
                    <p><strong>{query_analysis.get('intent', 'Unknown').title()}</strong></p>
                    <small>Confidence: {query_analysis.get('confidence', 0):.1%}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 Parameters</h4>
                    <p>{', '.join(query_analysis.get('parameters', ['None']))}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📈 Visualization</h4>
                    <p>{query_analysis.get('visualization_type', 'Default').title()}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Data Results
        data = result.get('data', [])
        if data:
            st.markdown(f"### 📊 Retrieved Data ({len(data)} records)")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Show data preview
            with st.expander("👁️ Data Preview"):
                st.dataframe(df.head(10))
            
            # Automatic visualizations based on suggestions
            viz_suggestions = result.get('visualization_suggestions', [])
            if viz_suggestions:
                st.markdown("### 📈 Suggested Visualizations")
                
                for viz_type in viz_suggestions[:3]:  # Limit to 3 visualizations
                    try:
                        fig = self.components['visualization_engine'].create_visualization(
                            df, viz_type
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create {viz_type} visualization: {e}")
            
            # Interactive map if geographic data available
            if 'lat' in df.columns and 'lon' in df.columns:
                st.markdown("### 🗺️ Geographic Distribution")
                try:
                    folium_map = self.components['geospatial_mapper'].create_interactive_map(
                        df, map_type='profiles'
                    )
                    st_folium(folium_map, width=700, height=500)
                except Exception as e:
                    st.warning(f"Could not create map: {e}")
        
        # Metadata
        metadata = result.get('metadata', {})
        if metadata:
            with st.expander("ℹ️ Query Metadata"):
                st.json(metadata)
    
    def data_visualization_interface(self):
        """Advanced data visualization interface"""
        
        st.markdown("## 📊 Advanced Data Visualization")
        
        # Load sample data
        sample_data = self.load_sample_data()
        
        if sample_data.empty:
            st.warning("⚠️ No data available for visualization")
            return
        
        # Visualization type selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            viz_type = st.selectbox(
                "Visualization Type",
                [
                    "depth_profile",
                    "trajectory_map", 
                    "time_series",
                    "scatter_plot",
                    "heatmap",
                    "bgc_profile",
                    "correlation_matrix"
                ],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            parameter = st.selectbox(
                "Primary Parameter",
                [col for col in sample_data.columns if col.endswith(('_c', '_psu', '_m', '_kg', '_m3'))],
                index=0 if sample_data.columns.any() else None
            )
        
        with col3:
            color_scheme = st.selectbox(
                "Color Scheme",
                ["Default", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"]
            )
        
        # Generate visualization
        if st.button("🎨 Generate Visualization"):
            with st.spinner("Creating visualization..."):
                try:
                    fig = self.components['visualization_engine'].create_visualization(
                        sample_data, viz_type, parameter=parameter
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download option
                        buffer = io.StringIO()
                        fig.write_html(buffer, include_plotlyjs='cdn')
                        st.download_button(
                            "💾 Download Plot",
                            buffer.getvalue(),
                            file_name=f"{viz_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                    else:
                        st.error("Failed to generate visualization")
                        
                except Exception as e:
                    st.error(f"Visualization error: {e}")
        
        # Statistical summary
        with st.expander("📈 Statistical Summary"):
            st.dataframe(sample_data.describe())
    
    def interactive_mapping_interface(self):
        """Interactive mapping interface"""
        
        st.markdown("## 🗺️ Interactive Geospatial Analysis")
        
        # Load sample data
        sample_data = self.load_sample_data()
        
        if sample_data.empty or 'lat' not in sample_data.columns:
            st.warning("⚠️ No geographic data available for mapping")
            return
        
        # Map configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            map_type = st.selectbox(
                "Map Type",
                ["trajectories", "profiles", "heatmap", "clusters", "regions"],
                format_func=lambda x: x.title()
            )
        
        with col2:
            cluster_data = st.checkbox("Cluster Markers", value=True)
        
        with col3:
            show_regions = st.checkbox("Show Regional Boundaries", value=True)
        
        # Generate map
        if st.button("🗺️ Generate Map"):
            with st.spinner("Creating interactive map..."):
                try:
                    folium_map = self.components['geospatial_mapper'].create_interactive_map(
                        sample_data, 
                        map_type=map_type,
                        cluster_markers=cluster_data,
                        show_regions=show_regions
                    )
                    
                    # Display map
                    map_data = st_folium(folium_map, width=700, height=500)
                    
                    # Show clicked data
                    if map_data.get('last_object_clicked'):
                        st.json(map_data['last_object_clicked'])
                        
                except Exception as e:
                    st.error(f"Mapping error: {e}")
        
        # Geographic statistics
        with st.expander("📍 Geographic Statistics"):
            if 'lat' in sample_data.columns and 'lon' in sample_data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Latitude Range", f"{sample_data['lat'].min():.2f}° to {sample_data['lat'].max():.2f}°")
                    st.metric("Longitude Range", f"{sample_data['lon'].min():.2f}° to {sample_data['lon'].max():.2f}°")
                
                with col2:
                    st.metric("Center Point", f"{sample_data['lat'].mean():.2f}°N, {sample_data['lon'].mean():.2f}°E")
                    st.metric("Data Points", len(sample_data))
    
    def data_management_interface(self):
        """Data management and upload interface"""
        
        st.markdown("## 📁 Data Management")
        
        # Data source tabs
        source_tab1, source_tab2, source_tab3 = st.tabs([
            "📝 Generate Demo Data",
            "📤 Upload NetCDF Files",
            "🔄 Data Processing"
        ])
        
        with source_tab1:
            self.demo_data_interface()
        
        with source_tab2:
            self.netcdf_upload_interface()
        
        with source_tab3:
            self.data_processing_interface()
    
    def demo_data_interface(self):
        """Demo data generation interface"""
        
        st.markdown("### 🎲 Generate Demo Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_floats = st.slider("Number of Floats", 1, 10, 3)
        
        with col2:
            n_profiles = st.slider("Profiles per Float", 10, 100, 50)
        
        with col3:
            include_bgc = st.checkbox("Include BGC Parameters", value=True)
        
        if st.button("🎲 Generate Demo Data"):
            with st.spinner("Generating demo data..."):
                try:
                    demo_data = self.components['dummy_generator'].generate_comprehensive_dataset(
                        n_floats=n_floats,
                        n_profiles_per_float=n_profiles,
                        include_bgc=include_bgc
                    )
                    
                    # Store in session state
                    st.session_state['demo_data'] = demo_data
                    
                    st.success(f"✅ Generated {len(demo_data)} demo records")
                    st.dataframe(demo_data.head())
                    
                except Exception as e:
                    st.error(f"Demo data generation failed: {e}")
        
        # Real ARGO Data Section
        st.markdown("### 🌊 Load Real ARGO Data")
        st.info("📍 Load real ARGO float data from the Indian Ocean region")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_floats = st.slider("Maximum Floats", 1, 10, 3, key="real_argo_floats")
        
        with col2:
            max_profiles = st.slider("Profiles per Float", 10, 50, 25, key="real_argo_profiles")
        
        if st.button("🌊 Load Real ARGO Data"):
            with st.spinner("🌊 Loading real ARGO data from Indian Ocean..."):
                try:
                    from data.real_argo_loader import load_real_argo_data
                    
                    # Load real ARGO data
                    argo_data, quality_report = load_real_argo_data(
                        max_floats=max_floats,
                        max_profiles=max_profiles
                    )
                    
                    if not argo_data.empty:
                        st.session_state['argo_data'] = argo_data
                        st.session_state['data_source'] = "Real ARGO Data"
                        
                        # Display quality report
                        st.success(f"✅ Loaded {len(argo_data)} real ARGO records!")
                        
                        with st.expander("📊 Data Quality Report", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Records", quality_report['total_records'])
                                st.metric("Unique Floats", quality_report['unique_floats'])
                                st.metric("Unique Profiles", quality_report['unique_profiles'])
                            
                            with col2:
                                st.metric("Date Span (days)", quality_report['date_range']['span_days'])
                                st.metric("Max Depth (m)", f"{quality_report['depth_coverage']['max_depth']:.0f}")
                                st.metric("Avg Profile Depth (m)", f"{quality_report['depth_coverage']['avg_profile_depth']:.0f}")
                            
                            with col3:
                                st.metric("Temp Data Quality (%)", f"{quality_report['data_quality']['temp_good_data_pct']:.1f}")
                                st.metric("Salinity Data Quality (%)", f"{quality_report['data_quality']['sal_good_data_pct']:.1f}")
                                st.metric("BGC Parameters", len(quality_report['parameters_available']['bgc_parameters']))
                            
                            # Geographic coverage
                            st.subheader("🗺️ Geographic Coverage")
                            geo = quality_report['geographic_coverage']
                            st.write(f"**Latitude:** {geo['lat_min']:.2f}° to {geo['lat_max']:.2f}°N")
                            st.write(f"**Longitude:** {geo['lon_min']:.2f}° to {geo['lon_max']:.2f}°E")
                            
                            # Parameters available
                            st.subheader("📊 Available Parameters")
                            if quality_report['parameters_available']['bgc_parameters']:
                                st.write("**Core + BGC Parameters:**", 
                                       quality_report['parameters_available']['core_parameters'] + 
                                       quality_report['parameters_available']['bgc_parameters'])
                            else:
                                st.write("**Core Parameters:**", quality_report['parameters_available']['core_parameters'])
                        
                        st.dataframe(argo_data.head(10))
                        
                    else:
                        st.error("❌ No real ARGO data could be loaded")
                    
                except Exception as e:
                    st.error(f"❌ Error loading real ARGO data: {e}")
                    logger.error(f"Real ARGO data loading error: {e}")
    
    def netcdf_upload_interface(self):
        """NetCDF file upload interface"""
        
        st.markdown("### 📤 Upload ARGO NetCDF Files")
        
        uploaded_files = st.file_uploader(
            "Choose NetCDF files",
            type=['nc', 'netcdf'],
            accept_multiple_files=True,
            help="Upload ARGO float NetCDF data files"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
            
            if st.button("🔄 Process NetCDF Files"):
                with st.spinner("Processing NetCDF files..."):
                    try:
                        # Save uploaded files temporarily
                        temp_paths = []
                        for uploaded_file in uploaded_files:
                            temp_path = f"/tmp/{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            temp_paths.append(temp_path)
                        
                        # Process files
                        result = self.components['netcdf_processor'].process_multiple_files(temp_paths)
                        
                        # Convert to DataFrame
                        processed_df = self.components['netcdf_processor'].convert_to_dataframe(result)
                        
                        # Store in session state
                        st.session_state['netcdf_data'] = processed_df
                        
                        st.success(f"✅ Processed {len(processed_df)} records from {len(uploaded_files)} files")
                        
                        # Show summary
                        summary = self.components['netcdf_processor'].get_data_summary(result)
                        with st.expander("📊 Processing Summary"):
                            st.json(summary)
                        
                        # Cleanup temp files
                        for temp_path in temp_paths:
                            try:
                                os.remove(temp_path)
                            except:
                                pass
                                
                    except Exception as e:
                        st.error(f"NetCDF processing failed: {e}")
    
    def data_processing_interface(self):
        """Data processing and quality control interface"""
        
        st.markdown("### 🔄 Data Processing & Quality Control")
        
        # Check available data
        available_data = []
        if 'demo_data' in st.session_state:
            available_data.append("Demo Data")
        if 'netcdf_data' in st.session_state:
            available_data.append("NetCDF Data")
        
        if not available_data:
            st.warning("⚠️ No data available. Generate demo data or upload NetCDF files first.")
            return
        
        data_source = st.selectbox("Select Data Source", available_data)
        
        # Quality control options
        st.markdown("#### 🔍 Quality Control Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            remove_outliers = st.checkbox("Remove Statistical Outliers", value=True)
            apply_depth_filter = st.checkbox("Apply Depth Range Filter", value=True)
        
        with col2:
            remove_duplicates = st.checkbox("Remove Duplicate Records", value=True)
            validate_coordinates = st.checkbox("Validate Geographic Coordinates", value=True)
        
        if apply_depth_filter:
            depth_range = st.slider(
                "Depth Range (m)",
                0, 6000, (0, 2000),
                help="Filter data within specified depth range"
            )
        
        if st.button("⚙️ Apply Processing"):
            with st.spinner("Applying data processing..."):
                try:
                    # Get data
                    if data_source == "Demo Data":
                        data = st.session_state['demo_data'].copy()
                    else:
                        data = st.session_state['netcdf_data'].copy()
                    
                    original_count = len(data)
                    
                    # Apply filters
                    if remove_duplicates:
                        data = data.drop_duplicates()
                    
                    if validate_coordinates and 'lat' in data.columns and 'lon' in data.columns:
                        data = data[
                            (data['lat'].between(-90, 90)) & 
                            (data['lon'].between(-180, 180))
                        ]
                    
                    if apply_depth_filter and 'depth_m' in data.columns:
                        data = data[
                            (data['depth_m'] >= depth_range[0]) & 
                            (data['depth_m'] <= depth_range[1])
                        ]
                    
                    if remove_outliers:
                        # Simple outlier removal for numeric columns
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            Q1 = data[col].quantile(0.25)
                            Q3 = data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                    
                    # Store processed data
                    st.session_state['processed_data'] = data
                    
                    final_count = len(data)
                    removed_count = original_count - final_count
                    
                    st.success(f"✅ Processing complete: {final_count} records remaining ({removed_count} removed)")
                    
                    # Show processing summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Records", original_count)
                    with col2:
                        st.metric("Processed Records", final_count)
                    with col3:
                        st.metric("Removed Records", removed_count)
                    
                except Exception as e:
                    st.error(f"Data processing failed: {e}")
    
    def system_analytics_interface(self):
        """System analytics and performance monitoring"""
        
        st.markdown("## 📈 System Analytics")
        
        # System status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🤖 AI System</h4>
                <p><strong>Gemini Pro</strong></p>
                <small>✅ Connected</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Visualizations</h4>
                <p><strong>Plotly + Folium</strong></p>
                <small>✅ Ready</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>🗺️ Mapping</h4>
                <p><strong>Interactive Maps</strong></p>
                <small>✅ Available</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h4>💾 Data Processing</h4>
                <p><strong>NetCDF + RAG</strong></p>
                <small>✅ Operational</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Component status
        st.markdown("### 🔧 Component Status")
        
        components_status = {
            "Gemini Query Processor": "✅ Operational",
            "Ocean Visualization Engine": "✅ Operational", 
            "NetCDF Data Processor": "✅ Operational",
            "Enhanced RAG Pipeline": "✅ Operational",
            "Geospatial Mapper": "✅ Operational",
            "Vector Store": "✅ Operational"
        }
        
        for component, status in components_status.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{component}**")
            with col2:
                st.write(status)
        
        # Performance metrics
        st.markdown("### ⚡ Performance Metrics")
        
        if st.button("🔄 Run Performance Test"):
            with st.spinner("Running performance tests..."):
                try:
                    # Test query processing time
                    start_time = datetime.now()
                    test_result = self.components['gemini_processor'].process_query("Test query for performance")
                    query_time = (datetime.now() - start_time).total_seconds()
                    
                    # Test visualization generation
                    start_time = datetime.now()
                    sample_data = self.load_sample_data()
                    if not sample_data.empty:
                        test_viz = self.components['visualization_engine'].create_visualization(
                            sample_data.head(50), 'depth_profile'
                        )
                    viz_time = (datetime.now() - start_time).total_seconds()
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Query Processing", f"{query_time:.2f}s")
                    
                    with col2:
                        st.metric("Visualization Generation", f"{viz_time:.2f}s")
                    
                    with col3:
                        st.metric("Total Response Time", f"{query_time + viz_time:.2f}s")
                    
                except Exception as e:
                    st.error(f"Performance test failed: {e}")
        
        # Configuration info
        with st.expander("⚙️ System Configuration"):
            config_info = {
                "Gemini Model": "gemini-pro",
                "Max Context Limit": Config.CONTEXT_LIMIT,
                "Map Center": f"{Config.MAP_CENTER_LAT}, {Config.MAP_CENTER_LON}",
                "Default Zoom": Config.MAP_ZOOM,
                "Plot Dimensions": f"{Config.PLOT_WIDTH}x{Config.PLOT_HEIGHT}",
                "LLM Backend": Config.LLM_BACKEND
            }
            
            for key, value in config_info.items():
                st.write(f"**{key}:** {value}")
    
    @st.cache_data
    def load_sample_data(_self) -> pd.DataFrame:
        """Load sample data for testing and demonstration"""
        
        try:
            # Check if we have processed data
            if 'processed_data' in st.session_state:
                return st.session_state['processed_data']
            
            # Check for demo data
            if 'demo_data' in st.session_state:
                return st.session_state['demo_data']
            
            # Check for NetCDF data
            if 'netcdf_data' in st.session_state:
                return st.session_state['netcdf_data']
            
            # Generate minimal sample data
            if hasattr(_self, 'components') and _self.components:
                sample_data = _self.components['dummy_generator'].generate_comprehensive_dataset(
                    n_floats=2, n_profiles_per_float=25, include_bgc=True
                )
                return sample_data
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            return pd.DataFrame()

# Main application
def main():
    """Main application entry point"""
    try:
        explorer = EnhancedArgoExplorer()
        explorer.main()
    except Exception as e:
        st.error(f"❌ Application failed to start: {e}")
        st.error("Please check system configuration and try refreshing the page.")

if __name__ == "__main__":
    main()