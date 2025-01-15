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
            # Highlight SQL keywords in removed lines
            highlighted = highlight_sql(line[1:])
            diff_lines.append(f'<div class="diff-remove"><span class="line-number"></span>{highlighted}</div>')
        elif line.startswith('+'):
            # Highlight SQL keywords in added lines
            highlighted = highlight_sql(line[1:])
            diff_lines.append(f'<div class="diff-add"><span class="line-number"></span>{highlighted}</div>')
        elif line.startswith('@@'):
            diff_lines.append(f'<div class="diff-info">{line}</div>')
        else:
            # Highlight SQL keywords in unchanged lines
            highlighted = highlight_sql(line)
            diff_lines.append(f'<div class="diff-unchanged"><span class="line-number"></span>{highlighted}</div>')
    
    diff_html = '\n'.join(diff_lines)
    logging.debug("Query diff created successfully")
    return diff_html

def highlight_sql(text: str) -> str:
    """Highlight SQL keywords in text.
    
    Args:
        text: SQL text to highlight
        
    Returns:
        HTML-formatted text with highlighted keywords
    """
    # Common SQL keywords to highlight
    keywords = {
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'JOIN',
        'LEFT', 'RIGHT', 'INNER', 'OUTER', 'ON', 'AND', 'OR', 'IN', 'NOT',
        'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'TABLE',
        'INDEX', 'VIEW', 'FUNCTION', 'PROCEDURE', 'TRIGGER', 'AS', 'CASE',
        'WHEN', 'THEN', 'ELSE', 'END', 'UNION', 'ALL', 'DISTINCT', 'TOP',
        'LIMIT', 'OFFSET', 'WITH', 'VALUES', 'INTO', 'NULL', 'IS', 'ASC',
        'DESC', 'BETWEEN', 'LIKE', 'EXISTS'
    }
    
    # Split into words while preserving whitespace and punctuation
    parts = []
    current_word = []
    for char in text:
        if char.isalnum() or char == '_':
            current_word.append(char)
        else:
            if current_word:
                word = ''.join(current_word)
                if word.upper() in keywords:
                    parts.append(f'<span class="keyword">{word}</span>')
                else:
                    parts.append(word)
                current_word = []
            parts.append(char)
    
    if current_word:
        word = ''.join(current_word)
        if word.upper() in keywords:
            parts.append(f'<span class="keyword">{word}</span>')
        else:
            parts.append(word)
    
    return ''.join(parts)

def display_query_comparison(original: str, optimized: str):
    """Display a side-by-side comparison of original and optimized queries."""
    st.markdown("### Query Comparison")
    
    # Create columns for side-by-side view
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Query**")
        st.code(format_sql(original), language="sql")
        
    with col2:
        st.markdown("**Optimized Query**")
        st.code(format_sql(optimized), language="sql")
    
    # Show diff below instead of in expander
    st.markdown("### Changes")
    st.markdown("""
    <style>
    .diff-legend {
        display: flex;
        gap: 20px;
        margin-bottom: 10px;
        font-family: monospace;
    }
    .diff-legend span {
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .diff-added { background-color: #2ea04326; }
    .diff-removed { background-color: #f8514926; }
    </style>
    <div class="diff-legend">
        <span><span style="color: #2ea043">+</span> Added</span>
        <span><span style="color: #f85149">-</span> Removed</span>
    </div>
    """, unsafe_allow_html=True)
    
    diff_html = create_query_diff(original, optimized)
    st.markdown(diff_html, unsafe_allow_html=True)

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


def render_advanced_optimization_view(analyzer: QueryAnalyzer):
    """Render the advanced optimization view."""
    st.markdown("## Advanced Optimization Mode")
    
    # Input section
    st.markdown("### Query Input")
    query = st.text_area("Enter your SQL query", height=200, key="advanced_sql_input")
    
    col1, col2 = st.columns(2)
    with col1:
        schema_info = st.text_area("Table Schema Information (Optional)", 
                                 placeholder="Enter table definitions, indexes, etc.",
                                 height=150)
    with col2:
        partition_info = st.text_area("Partitioning Details (Optional)",
                                   placeholder="Enter partitioning strategy details",
                                   height=150)
    
    # Analysis options
    st.markdown("### Optimization Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        analyze_clustering = st.checkbox("Analyze Clustering Keys", value=True)
        suggest_materialization = st.checkbox("Suggest Materialization", value=True)
    with col2:
        analyze_search = st.checkbox("Analyze Search Optimization", value=True)
        suggest_caching = st.checkbox("Suggest Caching Strategy", value=True)
    with col3:
        analyze_partitioning = st.checkbox("Analyze Partitioning", value=True)
    
    if st.button("Analyze Query"):
        if query:
            try:
                with st.spinner("Analyzing query..."):
                    result = analyzer.analyze_query(
                        query,
                        schema_info=schema_info if schema_info else None,
                        partition_info=partition_info if partition_info else None,
                        analyze_clustering=analyze_clustering,
                        suggest_materialization=suggest_materialization,
                        analyze_search=analyze_search,
                        suggest_caching=suggest_caching,
                        analyze_partitioning=analyze_partitioning
                    )
                    
                    if result:
                        st.markdown("### Analysis Results")
                        
                        # Query Information
                        st.markdown("#### Query Information")
                        st.markdown(f"**Category:** {result.category}")
                        st.markdown(f"**Confidence Score:** {result.confidence_score:.2f}")
                        
                        # Display comparisons
                        display_query_comparison(query, result.optimized_query)
                        
                        # Antipatterns
                        if result.antipatterns:
                            st.markdown("#### Detected Antipatterns")
                            for pattern in result.antipatterns:
                                st.warning(pattern)
                        
                        # Optimization suggestions
                        if result.suggestions:
                            st.markdown("#### Optimization Suggestions")
                            for suggestion in result.suggestions:
                                st.info(suggestion)
                    else:
                        st.error("Failed to analyze query. Please try again.")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
        else:
            st.warning("Please enter a SQL query to analyze.")


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