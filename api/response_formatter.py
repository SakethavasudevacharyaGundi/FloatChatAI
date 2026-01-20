"""
📊 ARGO Ocean Data Explorer - Response Formatting System
========================================================
Phase 11: Standardized Response Formatting

This module provides comprehensive response formatting for all API outputs,
ensuring consistent, rich, and extensible response structures across all
system components.

Features:
- Multi-format output support (JSON, HTML, XML, CSV)
- Rich metadata inclusion
- Performance metrics
- Error handling and user guidance
- Visualization integration
- Streaming response support
"""

import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import io
import base64
import logging

from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse, Response

logger = logging.getLogger(__name__)

class ResponseFormat(Enum):
    """Supported response formats."""
    JSON = "json"
    HTML = "html"
    XML = "xml"
    CSV = "csv"
    PLAIN_TEXT = "text"
    VISUALIZATION = "visualization"

class ResponseStatus(Enum):
    """Response status types."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    WARNING = "warning"
    ERROR = "error"
    PROCESSING = "processing"

class DataType(Enum):
    """Types of data in responses."""
    OCEANOGRAPHIC_DATA = "oceanographic_data"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    VISUALIZATION = "visualization"
    METADATA = "metadata"
    ERROR_INFO = "error_info"
    SYSTEM_INFO = "system_info"

class ResponseMetadata(BaseModel):
    """Standardized metadata for all responses."""
    timestamp: datetime
    processing_time: float
    query_id: str
    source_components: List[str]
    data_sources: List[str]
    confidence_score: Optional[float] = None
    cache_hit: bool = False
    performance_metrics: Dict[str, Any] = {}
    warnings: List[str] = []
    debug_info: Optional[Dict[str, Any]] = None

class DataSection(BaseModel):
    """Standardized data section structure."""
    type: DataType
    format: str
    content: Any
    metadata: Dict[str, Any] = {}
    schema_version: str = "1.0"

class OceanographicResponse(BaseModel):
    """Complete standardized response structure."""
    status: ResponseStatus
    message: str
    query: str
    data: List[DataSection] = []
    visualizations: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    metadata: ResponseMetadata
    links: Dict[str, str] = {}
    pagination: Optional[Dict[str, Any]] = None

class ResponseFormatter:
    """
    Comprehensive response formatting system for ARGO Ocean Data Explorer.
    
    Handles multiple output formats, rich metadata, and consistent structure
    across all API endpoints.
    """
    
    def __init__(self):
        self.format_handlers = {
            ResponseFormat.JSON: self._format_json,
            ResponseFormat.HTML: self._format_html,
            ResponseFormat.XML: self._format_xml,
            ResponseFormat.CSV: self._format_csv,
            ResponseFormat.PLAIN_TEXT: self._format_text,
            ResponseFormat.VISUALIZATION: self._format_visualization
        }
        
        # Response templates
        self.html_template = self._load_html_template()
        self.css_styles = self._load_css_styles()
        
        logger.info("Response Formatter initialized with multi-format support")
    
    def create_response(
        self,
        query: str,
        data: Any = None,
        status: ResponseStatus = ResponseStatus.SUCCESS,
        message: str = "",
        visualizations: List[Dict] = None,
        metadata: Dict[str, Any] = None,
        format_type: ResponseFormat = ResponseFormat.JSON,
        **kwargs
    ) -> Union[JSONResponse, HTMLResponse, Response]:
        """
        Create a standardized response in the specified format.
        
        Args:
            query: Original user query
            data: Response data
            status: Response status
            message: Human-readable message
            visualizations: List of visualization objects
            metadata: Additional metadata
            format_type: Desired response format
            **kwargs: Additional parameters
            
        Returns:
            Formatted response object
        """
        try:
            # Create metadata
            response_metadata = self._create_metadata(metadata or {}, **kwargs)
            
            # Structure data sections
            data_sections = self._structure_data_sections(data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(query, data, status)
            
            # Create complete response object
            response = OceanographicResponse(
                status=status,
                message=message or self._generate_default_message(status, query),
                query=query,
                data=data_sections,
                visualizations=visualizations or [],
                recommendations=recommendations,
                metadata=response_metadata,
                links=self._generate_links(query, format_type)
            )
            
            # Format according to requested type
            formatter = self.format_handlers.get(format_type, self._format_json)
            return formatter(response, **kwargs)
            
        except Exception as e:
            logger.error(f"Response formatting error: {str(e)}")
            # Return error response
            return self._create_error_response(str(e), query, format_type)
    
    def _create_metadata(self, base_metadata: Dict, **kwargs) -> ResponseMetadata:
        """Create comprehensive response metadata."""
        return ResponseMetadata(
            timestamp=datetime.now(),
            processing_time=kwargs.get("processing_time", 0.0),
            query_id=kwargs.get("query_id", f"query_{int(datetime.now().timestamp())}"),
            source_components=kwargs.get("source_components", []),
            data_sources=kwargs.get("data_sources", []),
            confidence_score=kwargs.get("confidence_score"),
            cache_hit=kwargs.get("cache_hit", False),
            performance_metrics=kwargs.get("performance_metrics", {}),
            warnings=kwargs.get("warnings", []),
            debug_info=kwargs.get("debug_info") if kwargs.get("include_debug", False) else None
        )
    
    def _structure_data_sections(self, data: Any) -> List[DataSection]:
        """Structure raw data into standardized sections."""
        if not data:
            return []
        
        sections = []
        
        if isinstance(data, dict):
            # Handle different types of data in the response
            for key, value in data.items():
                data_type = self._determine_data_type(key, value)
                section = DataSection(
                    type=data_type,
                    format="json",
                    content=value,
                    metadata={"source_key": key}
                )
                sections.append(section)
        
        elif isinstance(data, list):
            # Handle array data (likely oceanographic measurements)
            section = DataSection(
                type=DataType.OCEANOGRAPHIC_DATA,
                format="array",
                content=data,
                metadata={
                    "record_count": len(data),
                    "data_structure": "array"
                }
            )
            sections.append(section)
        
        else:
            # Handle single value or other types
            section = DataSection(
                type=DataType.OCEANOGRAPHIC_DATA,
                format="scalar",
                content=data,
                metadata={"data_type": type(data).__name__}
            )
            sections.append(section)
        
        return sections
    
    def _determine_data_type(self, key: str, value: Any) -> DataType:
        """Determine the type of data based on key and content."""
        key_lower = key.lower()
        
        if any(word in key_lower for word in ["temperature", "salinity", "depth", "pressure", "float"]):
            return DataType.OCEANOGRAPHIC_DATA
        elif any(word in key_lower for word in ["analysis", "statistics", "mean", "std", "correlation"]):
            return DataType.STATISTICAL_ANALYSIS
        elif any(word in key_lower for word in ["visualization", "chart", "plot", "graph"]):
            return DataType.VISUALIZATION
        elif any(word in key_lower for word in ["error", "exception", "warning"]):
            return DataType.ERROR_INFO
        elif any(word in key_lower for word in ["system", "health", "status", "performance"]):
            return DataType.SYSTEM_INFO
        else:
            return DataType.METADATA
    
    def _generate_recommendations(self, query: str, data: Any, status: ResponseStatus) -> List[str]:
        """Generate intelligent recommendations based on query and results."""
        recommendations = []
        query_lower = query.lower()
        
        # Query-based recommendations
        if "temperature" in query_lower and "profile" not in query_lower:
            recommendations.append("Consider viewing temperature profiles for depth-wise analysis")
        
        if "salinity" in query_lower and "temperature" not in query_lower:
            recommendations.append("A Temperature-Salinity diagram could provide additional insights")
        
        if any(word in query_lower for word in ["show", "display", "visualize"]) and not data:
            recommendations.append("Try a more specific query with location or time constraints")
        
        if "time" in query_lower or "trend" in query_lower:
            recommendations.append("Consider requesting a longer time series for better trend analysis")
        
        # Data-based recommendations
        if isinstance(data, list) and len(data) > 1000:
            recommendations.append("Large dataset returned - consider filtering for specific parameters")
        elif isinstance(data, list) and len(data) < 10:
            recommendations.append("Limited data found - try expanding search criteria")
        
        # Status-based recommendations
        if status == ResponseStatus.PARTIAL_SUCCESS:
            recommendations.append("Some data sources were unavailable - results may be incomplete")
        elif status == ResponseStatus.WARNING:
            recommendations.append("Check warnings in metadata for potential data quality issues")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _generate_links(self, query: str, format_type: ResponseFormat) -> Dict[str, str]:
        """Generate useful links for the response."""
        base_url = "/api"  # Could be made configurable
        
        links = {
            "self": f"{base_url}/query",
            "health": f"{base_url}/health",
            "visualize": f"{base_url}/visualize"
        }
        
        # Add format-specific links
        if format_type != ResponseFormat.JSON:
            links["json"] = f"{base_url}/query?format=json"
        if format_type != ResponseFormat.HTML:
            links["html"] = f"{base_url}/query?format=html"
        if format_type != ResponseFormat.CSV:
            links["csv"] = f"{base_url}/query?format=csv"
        
        return links
    
    def _generate_default_message(self, status: ResponseStatus, query: str) -> str:
        """Generate appropriate default message based on status."""
        messages = {
            ResponseStatus.SUCCESS: f"Successfully processed query: '{query[:50]}...'",
            ResponseStatus.PARTIAL_SUCCESS: f"Partially processed query with some limitations",
            ResponseStatus.WARNING: f"Query processed with warnings - check metadata",
            ResponseStatus.ERROR: f"Failed to process query: '{query[:50]}...'",
            ResponseStatus.PROCESSING: f"Processing query in progress..."
        }
        return messages.get(status, "Query processed")
    
    # Format-specific handlers
    
    def _format_json(self, response: OceanographicResponse, **kwargs) -> JSONResponse:
        """Format response as JSON."""
        return JSONResponse(
            content=response.dict(),
            headers={"Content-Type": "application/json"},
            status_code=200 if response.status == ResponseStatus.SUCCESS else 400
        )
    
    def _format_html(self, response: OceanographicResponse, **kwargs) -> HTMLResponse:
        """Format response as HTML."""
        html_content = self._generate_html_content(response)
        return HTMLResponse(
            content=html_content,
            headers={"Content-Type": "text/html"}
        )
    
    def _format_xml(self, response: OceanographicResponse, **kwargs) -> Response:
        """Format response as XML."""
        xml_content = self._generate_xml_content(response)
        return Response(
            content=xml_content,
            media_type="application/xml"
        )
    
    def _format_csv(self, response: OceanographicResponse, **kwargs) -> Response:
        """Format response as CSV (for tabular data)."""
        csv_content = self._generate_csv_content(response)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=oceanographic_data.csv"}
        )
    
    def _format_text(self, response: OceanographicResponse, **kwargs) -> Response:
        """Format response as plain text."""
        text_content = self._generate_text_content(response)
        return Response(
            content=text_content,
            media_type="text/plain"
        )
    
    def _format_visualization(self, response: OceanographicResponse, **kwargs) -> Response:
        """Format response with embedded visualizations."""
        if response.visualizations:
            # Return the first visualization if available
            viz = response.visualizations[0]
            if "html_content" in viz:
                return HTMLResponse(content=viz["html_content"])
            elif "file_path" in viz:
                with open(viz["file_path"], 'r') as f:
                    return HTMLResponse(content=f.read())
        
        # Fallback to HTML format
        return self._format_html(response, **kwargs)
    
    # Content generation methods
    
    def _generate_html_content(self, response: OceanographicResponse) -> str:
        """Generate HTML content for response."""
        # Status color mapping
        status_colors = {
            ResponseStatus.SUCCESS: "#28a745",
            ResponseStatus.PARTIAL_SUCCESS: "#ffc107", 
            ResponseStatus.WARNING: "#fd7e14",
            ResponseStatus.ERROR: "#dc3545",
            ResponseStatus.PROCESSING: "#17a2b8"
        }
        
        # Build data tables
        data_html = ""
        for section in response.data:
            if isinstance(section.content, list) and section.content:
                # Create table for array data
                data_html += f"<h3>{section.type.value.replace('_', ' ').title()}</h3>"
                data_html += self._create_html_table(section.content)
            else:
                # Show single values
                data_html += f"<div><strong>{section.type.value}:</strong> {section.content}</div>"
        
        # Build visualization section
        viz_html = ""
        if response.visualizations:
            viz_html = "<h3>Visualizations</h3>"
            for viz in response.visualizations:
                if "description" in viz:
                    viz_html += f"<p>{viz['description']}</p>"
                if "html_content" in viz:
                    viz_html += viz["html_content"]
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ARGO Ocean Data Explorer - Results</title>
    <style>{self.css_styles}</style>
</head>
<body>
    <div class="container">
        <h1>🌊 ARGO Ocean Data Explorer</h1>
        
        <div class="status-banner" style="background-color: {status_colors.get(response.status, '#6c757d')}">
            <strong>Status:</strong> {response.status.value.replace('_', ' ').title()}
        </div>
        
        <div class="section">
            <h2>Query</h2>
            <p class="query">"{response.query}"</p>
        </div>
        
        <div class="section">
            <h2>Message</h2>
            <p>{response.message}</p>
        </div>
        
        {f'<div class="section"><h2>Data</h2>{data_html}</div>' if data_html else ''}
        
        {f'<div class="section">{viz_html}</div>' if viz_html else ''}
        
        {f'<div class="section"><h2>Recommendations</h2><ul>{"".join([f"<li>{rec}</li>" for rec in response.recommendations])}</ul></div>' if response.recommendations else ''}
        
        <div class="metadata">
            <h3>Metadata</h3>
            <p><strong>Processing Time:</strong> {response.metadata.processing_time:.2f}s</p>
            <p><strong>Timestamp:</strong> {response.metadata.timestamp}</p>
            <p><strong>Components:</strong> {', '.join(response.metadata.source_components)}</p>
            {f'<p><strong>Confidence:</strong> {response.metadata.confidence_score:.2f}</p>' if response.metadata.confidence_score else ''}
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _create_html_table(self, data: List[Dict], max_rows: int = 50) -> str:
        """Create an HTML table from list of dictionaries."""
        if not data or not isinstance(data[0], dict):
            return f"<p>Data preview: {str(data[:5])}...</p>"
        
        # Get headers
        headers = list(data[0].keys())
        
        # Create table
        table_html = "<table class='data-table'>"
        table_html += "<thead><tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr></thead>"
        table_html += "<tbody>"
        
        for row in data[:max_rows]:
            table_html += "<tr>" + "".join([f"<td>{row.get(h, '')}</td>" for h in headers]) + "</tr>"
        
        table_html += "</tbody></table>"
        
        if len(data) > max_rows:
            table_html += f"<p><em>Showing {max_rows} of {len(data)} rows</em></p>"
        
        return table_html
    
    def _generate_xml_content(self, response: OceanographicResponse) -> str:
        """Generate XML content for response."""
        root = ET.Element("oceanographic_response")
        
        # Add basic info
        ET.SubElement(root, "status").text = response.status.value
        ET.SubElement(root, "message").text = response.message
        ET.SubElement(root, "query").text = response.query
        
        # Add data sections
        data_elem = ET.SubElement(root, "data")
        for section in response.data:
            section_elem = ET.SubElement(data_elem, "section")
            section_elem.set("type", section.type.value)
            section_elem.text = str(section.content)
        
        # Add metadata
        metadata_elem = ET.SubElement(root, "metadata")
        ET.SubElement(metadata_elem, "timestamp").text = response.metadata.timestamp.isoformat()
        ET.SubElement(metadata_elem, "processing_time").text = str(response.metadata.processing_time)
        
        return ET.tostring(root, encoding='unicode')
    
    def _generate_csv_content(self, response: OceanographicResponse) -> str:
        """Generate CSV content for tabular data."""
        output = io.StringIO()
        
        # Find tabular data
        tabular_data = None
        for section in response.data:
            if isinstance(section.content, list) and section.content:
                if isinstance(section.content[0], dict):
                    tabular_data = section.content
                    break
        
        if tabular_data:
            # Write CSV
            fieldnames = list(tabular_data[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tabular_data)
        else:
            # Write simple CSV with available data
            writer = csv.writer(output)
            writer.writerow(["Query", response.query])
            writer.writerow(["Status", response.status.value])
            writer.writerow(["Message", response.message])
            writer.writerow(["Timestamp", response.metadata.timestamp.isoformat()])
        
        return output.getvalue()
    
    def _generate_text_content(self, response: OceanographicResponse) -> str:
        """Generate plain text content."""
        lines = [
            "🌊 ARGO Ocean Data Explorer - Results",
            "=" * 50,
            f"Query: {response.query}",
            f"Status: {response.status.value}",
            f"Message: {response.message}",
            "",
            "Data:",
        ]
        
        for section in response.data:
            lines.append(f"  {section.type.value}: {str(section.content)[:100]}...")
        
        if response.recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in response.recommendations:
                lines.append(f"  • {rec}")
        
        lines.extend([
            "",
            f"Processing Time: {response.metadata.processing_time:.2f}s",
            f"Timestamp: {response.metadata.timestamp}"
        ])
        
        return "\n".join(lines)
    
    def _load_html_template(self) -> str:
        """Load HTML template for responses."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>ARGO Ocean Data Explorer</title>
    <style>{styles}</style>
</head>
<body>
    {content}
</body>
</html>"""
    
    def _load_css_styles(self) -> str:
        """Load CSS styles for HTML responses."""
        return """
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        .status-banner { padding: 15px; margin: 20px 0; border-radius: 5px; color: white; font-weight: bold; }
        .section { margin: 25px 0; padding: 20px; border-left: 4px solid #3498db; background: #f8f9fa; }
        .query { font-style: italic; font-size: 18px; color: #2c3e50; background: #ecf0f1; padding: 10px; border-radius: 5px; }
        .data-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .data-table th, .data-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        .data-table th { background: #3498db; color: white; }
        .data-table tr:nth-child(even) { background: #f2f2f2; }
        .metadata { background: #ecf0f1; padding: 20px; border-radius: 5px; margin-top: 30px; }
        ul li { margin: 5px 0; }
        """
    
    def _create_error_response(self, error: str, query: str, format_type: ResponseFormat) -> Response:
        """Create an error response."""
        error_response = OceanographicResponse(
            status=ResponseStatus.ERROR,
            message=f"Error processing request: {error}",
            query=query,
            metadata=ResponseMetadata(
                timestamp=datetime.now(),
                processing_time=0.0,
                query_id=f"error_{int(datetime.now().timestamp())}",
                source_components=["response_formatter"],
                data_sources=[]
            )
        )
        
        formatter = self.format_handlers.get(format_type, self._format_json)
        return formatter(error_response)

# Global formatter instance
response_formatter = ResponseFormatter()