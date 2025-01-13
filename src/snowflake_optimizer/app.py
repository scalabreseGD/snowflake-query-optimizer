"""Main Streamlit application for Snowflake Query Optimizer."""

import os
from typing import Optional, List, Dict
import json
import logging
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import sqlparse
import difflib

from snowflake_optimizer.data_collector import QueryMetricsCollector
from snowflake_optimizer.query_analyzer import QueryAnalyzer, SchemaInfo

# Configure logging
def setup_logging():
    """Configure logging with custom format and handlers."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"snowflake_optimizer_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Create formatters and handlers
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s'
    )
    
    # File handler for detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # Log initial application startup
    logging.info("Snowflake Query Optimizer application started")
    logging.debug(f"Log file created at: {log_file}")

# Set up logging
setup_logging()

# Load environment variables
load_dotenv()

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = None
if "selected_query" not in st.session_state:
    st.session_state.selected_query = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "manual_queries" not in st.session_state:
    st.session_state.manual_queries = []
if "schema_info" not in st.session_state:
    st.session_state.schema_info = None

def format_sql(query: str) -> str:
    """Format SQL query for better readability.
    
    Args:
        query: SQL query string
        
    Returns:
        Formatted SQL query
    """
    logging.debug(f"Formatting SQL query of length: {len(query)}")
    try:
        formatted = sqlparse.format(
            query,
            reindent=True,
            keyword_case='upper',
            identifier_case='lower',
            indent_width=4
        )
        logging.debug("SQL formatting successful")
        return formatted
    except Exception as e:
        logging.error(f"SQL formatting failed: {str(e)}")
        return query

def initialize_connections() -> tuple[Optional[QueryMetricsCollector], Optional[QueryAnalyzer]]:
    """Initialize connections to Snowflake and LLM services.

    Returns:
        Tuple of QueryMetricsCollector and QueryAnalyzer instances
    """
    logging.info("Initializing service connections")
    
    try:
        logging.debug("Attempting to connect to Snowflake")
        collector = QueryMetricsCollector(
            account=st.secrets["SNOWFLAKE_ACCOUNT"],
            user=st.secrets["SNOWFLAKE_USER"],
            password=st.secrets["SNOWFLAKE_PASSWORD"],
            warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
            database=st.secrets.get("SNOWFLAKE_DATABASE"),
            schema=st.secrets.get("SNOWFLAKE_SCHEMA"),
        )
        logging.info("Successfully connected to Snowflake")
    except Exception as e:
        logging.error(f"Failed to connect to Snowflake: {str(e)}")
        st.error(f"Failed to connect to Snowflake: {str(e)}")
        collector = None

    try:
        logging.debug("Initializing Query Analyzer")
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        logging.debug(f"API key length: {len(api_key)}")
        analyzer = QueryAnalyzer(
            anthropic_api_key=api_key
        )
        logging.info("Successfully initialized Query Analyzer")
    except Exception as e:
        logging.error(f"Failed to initialize Query Analyzer: {str(e)}")
        st.error(f"Failed to initialize Query Analyzer: {str(e)}")
        analyzer = None

    return collector, analyzer


def create_query_diff(original: str, optimized: str) -> str:
    """Create a diff view between original and optimized queries.
    
    Args:
        original: Original SQL query
        optimized: Optimized SQL query
        
    Returns:
        HTML-formatted diff string
    """
    logging.debug("Creating diff between original and optimized queries")
    
    # Format both queries
    original_formatted = format_sql(original).splitlines()
    optimized_formatted = format_sql(optimized).splitlines()
    
    # Create unified diff
    diff_lines = []
    for line in difflib.unified_diff(
        original_formatted,
        optimized_formatted,
        fromfile='Original',
        tofile='Optimized',
        lineterm='',
    ):
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('-'):
            diff_lines.append(f'<div class="diff-remove">{line[1:]}</div>')
        elif line.startswith('+'):
            diff_lines.append(f'<div class="diff-add">{line[1:]}</div>')
        elif line.startswith('@@'):
            diff_lines.append(f'<div class="diff-info">{line}</div>')
        else:
            diff_lines.append(f'<div class="diff-unchanged">{line}</div>')
    
    diff_html = '\n'.join(diff_lines)
    logging.debug("Query diff created successfully")
    return diff_html

def display_query_comparison(original: str, optimized: str):
    """Display side-by-side comparison of original and optimized queries.
    
    Args:
        original: Original SQL query
        optimized: Optimized SQL query
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Original Query")
        st.code(format_sql(original), language="sql")
        
    with col2:
        st.markdown("### Optimized Query")
        st.code(format_sql(optimized), language="sql")
    
    # Show diff view in expander
    with st.expander("View Changes"):
        st.markdown("""
        <style>
            .diff-view {
                font-family: 'JetBrains Mono', 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.5;
                background-color: #f8f9fa;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid #e9ecef;
                overflow-x: auto;
            }
            .diff-remove {
                background-color: #ffeef0;
                color: #b31d28;
                padding: 2px 4px;
                margin: 2px 0;
                border-radius: 4px;
                position: relative;
            }
            .diff-remove::before {
                content: "−";
                color: #b31d28;
                margin-right: 8px;
                font-weight: bold;
            }
            .diff-add {
                background-color: #e6ffec;
                color: #22863a;
                padding: 2px 4px;
                margin: 2px 0;
                border-radius: 4px;
                position: relative;
            }
            .diff-add::before {
                content: "+";
                color: #22863a;
                margin-right: 8px;
                font-weight: bold;
            }
            .diff-unchanged {
                color: #24292e;
                padding: 2px 4px;
                margin: 2px 0;
            }
            .diff-unchanged::before {
                content: " ";
                margin-right: 8px;
                opacity: 0.3;
            }
            .diff-info {
                color: #6a737d;
                padding: 2px 4px;
                margin: 4px 0;
                font-style: italic;
                border-top: 1px solid #e1e4e8;
                border-bottom: 1px solid #e1e4e8;
            }
        </style>
        <div class="diff-view">
        """, unsafe_allow_html=True)
        
        diff_html = create_query_diff(original, optimized)
        st.markdown(diff_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add a legend
        st.markdown("""
        <div style="margin-top: 16px; font-size: 14px;">
            <span style="color: #22863a;">●</span> Added &nbsp;&nbsp;
            <span style="color: #b31d28;">●</span> Removed &nbsp;&nbsp;
            <span style="color: #24292e;">●</span> Unchanged
        </div>
        """, unsafe_allow_html=True)

def render_query_history_view(collector: Optional[QueryMetricsCollector], analyzer: Optional[QueryAnalyzer]):
    """Render the query history analysis view.

    Args:
        collector: QueryMetricsCollector instance
        analyzer: QueryAnalyzer instance
    """
    logging.info("Rendering query history view")
    st.header("Query History Analysis")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("History Configuration")
        days = st.slider(
            "Days to analyze",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to look back in query history"
        )
        min_execution_time = st.number_input(
            "Minimum execution time (seconds)",
            min_value=1,
            value=60,
            help="Minimum query execution time to consider"
        )
        limit = st.number_input(
            "Number of queries",
            min_value=1,
            max_value=1000,
            value=100,
            help="Maximum number of queries to analyze"
        )
        
        logging.debug(f"Query history parameters - days: {days}, min_execution_time: {min_execution_time}, limit: {limit}")

        if st.button("Fetch Queries"):
            if collector:
                logging.info("Fetching query history from Snowflake")
                with st.spinner("Fetching query history..."):
                    try:
                        st.session_state.query_history = collector.get_expensive_queries(
                            days=days,
                            min_execution_time=min_execution_time,
                            limit=limit
                        )
                        logging.info(f"Successfully fetched {len(st.session_state.query_history)} queries")
                    except Exception as e:
                        logging.error(f"Failed to fetch query history: {str(e)}")
                        st.error(f"Failed to fetch queries: {str(e)}")
            else:
                logging.error("Cannot fetch queries - Snowflake connection not available")
                st.error("Snowflake connection not available")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.session_state.query_history is not None:
            st.dataframe(
                st.session_state.query_history[[
                    "QUERY_ID",
                    "EXECUTION_TIME_SECONDS",
                    "MB_SCANNED",
                    "ROWS_PRODUCED"
                ]],
                height=400
            )

            selected_query_id = st.selectbox(
                "Select a query to analyze",
                st.session_state.query_history["QUERY_ID"].tolist()
            )
            
            if selected_query_id:
                logging.debug(f"Selected query ID: {selected_query_id}")
                query_text = st.session_state.query_history[
                    st.session_state.query_history["QUERY_ID"] == selected_query_id
                ]["QUERY_TEXT"].iloc[0]
                st.session_state.selected_query = format_sql(query_text)
                logging.info(f"Query loaded for analysis - ID: {selected_query_id}")

    with col2:
        if st.session_state.selected_query:
            st.markdown("### Selected Query")
            st.code(st.session_state.selected_query, language="sql")

            if st.button("Analyze Query"):
                if analyzer:
                    logging.info("Starting query analysis")
                    with st.spinner("Analyzing query..."):
                        try:
                            st.session_state.analysis_results = analyzer.analyze_query(
                                st.session_state.selected_query
                            )
                            logging.info("Query analysis completed successfully")
                        except Exception as e:
                            logging.error(f"Query analysis failed: {str(e)}")
                            st.error(f"Analysis failed: {str(e)}")

        if st.session_state.analysis_results:
            st.subheader("Analysis Results")
            
            # Log analysis results
            logging.debug(f"Analysis results - Category: {st.session_state.analysis_results.category}, "
                         f"Complexity: {st.session_state.analysis_results.complexity_score:.2f}")
            
            # Display query category and complexity
            st.info(f"Query Category: {st.session_state.analysis_results.category}")
            st.progress(st.session_state.analysis_results.complexity_score, 
                       text=f"Complexity Score: {st.session_state.analysis_results.complexity_score:.2f}")
            
            # Display antipatterns
            if st.session_state.analysis_results.antipatterns:
                logging.debug(f"Antipatterns detected: {len(st.session_state.analysis_results.antipatterns)}")
                st.warning("Antipatterns Detected:")
                for pattern in st.session_state.analysis_results.antipatterns:
                    st.write(f"- {pattern}")

            # Display suggestions
            if st.session_state.analysis_results.suggestions:
                logging.debug(f"Optimization suggestions: {len(st.session_state.analysis_results.suggestions)}")
                st.info("Optimization Suggestions:")
                for suggestion in st.session_state.analysis_results.suggestions:
                    st.write(f"- {suggestion}")

            # Display optimized query
            if st.session_state.analysis_results.optimized_query:
                logging.info("Optimized query generated")
                st.success("Optimized Query:")
                formatted_query = format_sql(st.session_state.analysis_results.optimized_query)
                st.code(formatted_query, language="sql")
                if st.button("Copy Optimized Query"):
                    st.session_state.clipboard = formatted_query
                    logging.debug("Optimized query copied to clipboard")
                    st.success("Query copied to clipboard!")


def render_manual_analysis_view(analyzer: Optional[QueryAnalyzer]):
    """Render the manual query analysis view.

    Args:
        analyzer: QueryAnalyzer instance
    """
    st.header("Manual Query Analysis")
    
    # Initialize session state for formatted query
    if "formatted_query" not in st.session_state:
        st.session_state.formatted_query = ""
    
    # Query input methods
    input_method = st.radio(
        "Choose input method",
        ["Direct Input", "File Upload", "Batch Analysis"]
    )
    
    if input_method == "Direct Input":
        st.markdown("### Enter SQL Query")
        
        # Format button before text area
        if st.button("Format Query"):
            if st.session_state.formatted_query:
                st.session_state.formatted_query = format_sql(st.session_state.formatted_query)
                logging.debug("Query formatted successfully")
        
        # Text area for SQL input
        query = st.text_area(
            "SQL Query",
            value=st.session_state.formatted_query,
            height=200,
            help="Paste your SQL query here for analysis",
            key="sql_input",
            on_change=lambda: setattr(st.session_state, 'formatted_query', st.session_state.sql_input)
        )
        
        # Display formatted query with syntax highlighting
        if query:
            st.markdown("### Preview")
            st.code(query, language="sql")
        
        # Optional schema information
        if st.checkbox("Add table schema information"):
            schema_col1, schema_col2 = st.columns([1, 1])
            
            with schema_col1:
                table_name = st.text_input("Table name")
                row_count = st.number_input("Approximate row count", min_value=0)
            
            with schema_col2:
                columns_json = st.text_area(
                    "Columns (JSON format)",
                    help='Example: [{"name": "id", "type": "INTEGER"}, {"name": "email", "type": "VARCHAR"}]'
                )
                
            try:
                columns = json.loads(columns_json) if columns_json else []
                st.session_state.schema_info = SchemaInfo(
                    table_name=table_name,
                    columns=columns,
                    row_count=row_count
                )
                logging.debug(f"Schema info updated - Table: {table_name}, Columns: {len(columns)}")
            except json.JSONDecodeError:
                logging.error("Invalid JSON format provided for columns")
                st.error("Invalid JSON format for columns")
                st.session_state.schema_info = None
        
        if st.button("Analyze"):
            if query and analyzer:
                logging.info("Starting manual query analysis")
                with st.spinner("Analyzing query..."):
                    try:
                        st.session_state.analysis_results = analyzer.analyze_query(
                            query,
                            schema_info=st.session_state.schema_info
                        )
                        logging.info("Manual query analysis completed successfully")
                    except Exception as e:
                        logging.error(f"Manual query analysis failed: {str(e)}")
                        st.error(f"Analysis failed: {str(e)}")
                    
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader("Upload SQL file", type=["sql"])
        if uploaded_file and analyzer:
            try:
                query = uploaded_file.getvalue().decode()
                st.markdown("### SQL Query")
                formatted_query = format_sql(query)
                st.code(formatted_query, language="sql")
                logging.info(f"SQL file uploaded successfully: {uploaded_file.name}")
                
                if st.button("Analyze"):
                    logging.info("Starting uploaded file analysis")
                    with st.spinner("Analyzing query..."):
                        try:
                            st.session_state.analysis_results = analyzer.analyze_query(formatted_query)
                            logging.info("File analysis completed successfully")
                        except Exception as e:
                            logging.error(f"File analysis failed: {str(e)}")
                            st.error(f"Analysis failed: {str(e)}")
            except Exception as e:
                logging.error(f"Failed to read uploaded file: {str(e)}")
                st.error(f"Failed to read file: {str(e)}")
                    
    else:  # Batch Analysis
        uploaded_files = st.file_uploader(
            "Upload SQL files",
            type=["sql"],
            accept_multiple_files=True
        )
        
        if uploaded_files and analyzer:
            if st.button("Analyze All"):
                results = []
                progress_bar = st.progress(0)
                logging.info(f"Starting batch analysis of {len(uploaded_files)} files")
                
                for i, file in enumerate(uploaded_files):
                    try:
                        query = file.getvalue().decode()
                        st.markdown(f"### Query from {file.name}")
                        formatted_query = format_sql(query)
                        st.code(formatted_query, language="sql")
                        
                        analysis = analyzer.analyze_query(formatted_query)
                        results.append({
                            "filename": file.name,
                            "analysis": analysis
                        })
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        logging.info(f"Analyzed file {i+1}/{len(uploaded_files)}: {file.name}")
                    except Exception as e:
                        logging.error(f"Failed to analyze {file.name}: {str(e)}")
                        st.error(f"Failed to analyze {file.name}: {str(e)}")
                
                st.session_state.manual_queries = results
                logging.info("Batch analysis completed")
    
    # Display analysis results
    if st.session_state.analysis_results:
        st.subheader("Analysis Results")
        
        # Display query category and complexity
        st.info(f"Query Category: {st.session_state.analysis_results.category}")
        st.progress(
            st.session_state.analysis_results.complexity_score,
            text=f"Complexity Score: {st.session_state.analysis_results.complexity_score:.2f}"
        )
        
        # Display antipatterns
        if st.session_state.analysis_results.antipatterns:
            st.warning("Antipatterns Detected:")
            for pattern in st.session_state.analysis_results.antipatterns:
                st.write(f"- {pattern}")
        
        # Display suggestions
        if st.session_state.analysis_results.suggestions:
            st.info("Optimization Suggestions:")
            for suggestion in st.session_state.analysis_results.suggestions:
                st.write(f"- {suggestion}")
        
        # Display optimized query
        if st.session_state.analysis_results.optimized_query:
            st.success("Query Optimization Results")
            display_query_comparison(
                query,
                st.session_state.analysis_results.optimized_query
            )
            if st.button("Copy Optimized Query"):
                st.session_state.clipboard = format_sql(st.session_state.analysis_results.optimized_query)
                logging.debug("Optimized query copied to clipboard")
                st.success("Query copied to clipboard!")
    
    # Display batch analysis results
    if st.session_state.manual_queries:
        st.subheader("Batch Analysis Results")
        for result in st.session_state.manual_queries:
            with st.expander(f"Results for {result['filename']}"):
                st.info(f"Query Category: {result['analysis'].category}")
                st.progress(
                    result['analysis'].complexity_score,
                    text=f"Complexity Score: {result['analysis'].complexity_score:.2f}"
                )
                
                if result['analysis'].antipatterns:
                    st.warning("Antipatterns:")
                    for pattern in result['analysis'].antipatterns:
                        st.write(f"- {pattern}")
                
                if result['analysis'].optimized_query:
                    st.success("Optimized Query:")
                    st.code(format_sql(result['analysis'].optimized_query), language="sql")
                    logging.debug(f"Displayed results for {result['filename']}")


def render_advanced_optimization_view(analyzer: Optional[QueryAnalyzer]):
    """Render the advanced Snowflake optimization view.

    Args:
        analyzer: QueryAnalyzer instance
    """
    st.header("Advanced Snowflake Optimization")
    
    # Input section
    st.markdown("### Enter SQL Query")
    query = st.text_area(
        "SQL Query",
        height=200,
        help="Paste your SQL query here for advanced Snowflake-specific optimizations",
        key="advanced_sql_input"
    )
    
    # Format button
    if query and st.button("Format Query"):
        formatted_query = format_sql(query)
        st.session_state.advanced_sql_input = formatted_query
        st.experimental_rerun()
    
    # Display formatted query with syntax highlighting
    if query:
        st.markdown("### Formatted Query")
        st.code(format_sql(query), language="sql")
    
    # Advanced configuration
    with st.expander("Advanced Configuration"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            enable_clustering = st.checkbox("Analyze Clustering Keys", value=True)
            enable_materialization = st.checkbox("Analyze Materialization", value=True)
            enable_search = st.checkbox("Analyze Search Optimization", value=True)
        
        with col2:
            enable_caching = st.checkbox("Analyze Caching Strategy", value=True)
            enable_partitioning = st.checkbox("Analyze Partitioning", value=True)
    
    # Schema information
    with st.expander("Table Schema Information"):
        table_name = st.text_input("Table name")
        row_count = st.number_input("Approximate row count", min_value=0)
        size_bytes = st.number_input("Size in bytes", min_value=0)
        
        st.subheader("Columns")
        col_name = st.text_input("Column name")
        col_type = st.text_input("Column type")
        if st.button("Add Column"):
            if not hasattr(st.session_state, 'columns'):
                st.session_state.columns = []
            if col_name and col_type:
                st.session_state.columns.append({"name": col_name, "type": col_type})
        
        if hasattr(st.session_state, 'columns'):
            for i, col in enumerate(st.session_state.columns):
                st.write(f"{i+1}. {col['name']} ({col['type']})")
            if st.button("Clear Columns"):
                st.session_state.columns = []
        
        # Partitioning information
        st.subheader("Partitioning")
        partition_col = st.text_input("Partition column")
        partition_type = st.selectbox(
            "Partition type",
            ["RANGE", "LIST", "HASH"]
        )
        if st.button("Set Partitioning"):
            st.session_state.partitioning = {
                "column": partition_col,
                "type": partition_type
            }
    
    if st.button("Analyze with Advanced Optimizations"):
        if query and analyzer:
            with st.spinner("Performing advanced analysis..."):
                # Create schema info object
                schema_info = None
                if hasattr(st.session_state, 'columns'):
                    schema_info = SchemaInfo(
                        table_name=table_name,
                        columns=st.session_state.columns,
                        row_count=row_count,
                        size_bytes=size_bytes,
                        partitioning=getattr(st.session_state, 'partitioning', None)
                    )
                
                # Get analysis results
                analysis_results = analyzer.analyze_query(
                    query,
                    schema_info=schema_info
                )
                
                # Display results in organized sections
                st.subheader("Analysis Results")
                
                # Query Information
                with st.expander("Query Information", expanded=True):
                    st.info(f"Query Category: {analysis_results.category}")
                    st.progress(
                        analysis_results.complexity_score,
                        text=f"Complexity Score: {analysis_results.complexity_score:.2f}"
                    )
                
                # Antipatterns
                if analysis_results.antipatterns:
                    with st.expander("Antipatterns Detected", expanded=True):
                        for pattern in analysis_results.antipatterns:
                            st.warning(f"• {pattern}")
                
                # Clustering Recommendations
                if enable_clustering and any("cluster" in s.lower() for s in analysis_results.suggestions):
                    with st.expander("Clustering Recommendations", expanded=True):
                        for suggestion in analysis_results.suggestions:
                            if "cluster" in suggestion.lower():
                                st.info(f"• {suggestion}")
                
                # Materialization Recommendations
                if enable_materialization and analysis_results.materialization_suggestions:
                    with st.expander("Materialization Recommendations", expanded=True):
                        for suggestion in analysis_results.materialization_suggestions:
                            st.info(f"• {suggestion}")
                
                # Search Optimization
                if enable_search and any("search" in s.lower() for s in analysis_results.suggestions):
                    with st.expander("Search Optimization Recommendations", expanded=True):
                        for suggestion in analysis_results.suggestions:
                            if "search" in suggestion.lower():
                                st.info(f"• {suggestion}")
                
                # Caching Strategy
                if enable_caching and any("cache" in s.lower() for s in analysis_results.suggestions):
                    with st.expander("Caching Recommendations", expanded=True):
                        for suggestion in analysis_results.suggestions:
                            if "cache" in suggestion.lower():
                                st.info(f"• {suggestion}")
                
                # Optimized Query
                if analysis_results.optimized_query:
                    with st.expander("Query Optimization Results", expanded=True):
                        display_query_comparison(
                            query,
                            analysis_results.optimized_query
                        )
                        if st.button("Copy Optimized Query"):
                            st.session_state.clipboard = format_sql(analysis_results.optimized_query)
                            st.success("Query copied to clipboard!")


def main():
    """Main function to run the Streamlit application."""
    logging.info("Starting main application")
    st.title("Snowflake Query Optimizer")
    st.write("Analyze and optimize your Snowflake SQL queries")

    # Initialize connections
    collector, analyzer = initialize_connections()

    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["Query History Analysis", "Manual Analysis", "Advanced Optimization"]
    )
    logging.info(f"Selected mode: {mode}")

    if mode == "Query History Analysis":
        render_query_history_view(collector, analyzer)
    elif mode == "Manual Analysis":
        render_manual_analysis_view(analyzer)
    else:
        render_advanced_optimization_view(analyzer)

    logging.debug("Main application loop completed")


if __name__ == "__main__":
    main() 