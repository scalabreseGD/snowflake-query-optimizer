"""Main Streamlit application for Snowflake Query Optimizer."""

import os
from typing import Optional, List, Dict
import json
import streamlit as st
from dotenv import load_dotenv
import sqlparse

from snowflake_optimizer.data_collector import QueryMetricsCollector
from snowflake_optimizer.query_analyzer import QueryAnalyzer, SchemaInfo

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
    try:
        return sqlparse.format(
            query,
            reindent=True,
            keyword_case='upper',
            identifier_case='lower',
            indent_width=4
        )
    except:
        return query

def initialize_connections() -> tuple[Optional[QueryMetricsCollector], Optional[QueryAnalyzer]]:
    """Initialize connections to Snowflake and LLM services.

    Returns:
        Tuple of QueryMetricsCollector and QueryAnalyzer instances
    """
    try:
        collector = QueryMetricsCollector(
            account=st.secrets["SNOWFLAKE_ACCOUNT"],
            user=st.secrets["SNOWFLAKE_USER"],
            password=st.secrets["SNOWFLAKE_PASSWORD"],
            warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
            database=st.secrets.get("SNOWFLAKE_DATABASE"),
            schema=st.secrets.get("SNOWFLAKE_SCHEMA"),
        )
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {str(e)}")
        collector = None

    try:
        analyzer = QueryAnalyzer(
            anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to initialize Query Analyzer: {str(e)}")
        analyzer = None

    return collector, analyzer


def render_query_history_view(collector: Optional[QueryMetricsCollector], analyzer: Optional[QueryAnalyzer]):
    """Render the query history analysis view.

    Args:
        collector: QueryMetricsCollector instance
        analyzer: QueryAnalyzer instance
    """
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

        if st.button("Fetch Queries"):
            if collector:
                with st.spinner("Fetching query history..."):
                    st.session_state.query_history = collector.get_expensive_queries(
                        days=days,
                        min_execution_time=min_execution_time,
                        limit=limit
                    )
            else:
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
                query_text = st.session_state.query_history[
                    st.session_state.query_history["QUERY_ID"] == selected_query_id
                ]["QUERY_TEXT"].iloc[0]
                st.session_state.selected_query = format_sql(query_text)

    with col2:
        if st.session_state.selected_query:
            st.markdown("### Selected Query")
            st.code(st.session_state.selected_query, language="sql")

            if st.button("Analyze Query"):
                if analyzer:
                    with st.spinner("Analyzing query..."):
                        st.session_state.analysis_results = analyzer.analyze_query(
                            st.session_state.selected_query
                        )

        if st.session_state.analysis_results:
            st.subheader("Analysis Results")
            
            # Display query category and complexity
            st.info(f"Query Category: {st.session_state.analysis_results.category}")
            st.progress(st.session_state.analysis_results.complexity_score, 
                       text=f"Complexity Score: {st.session_state.analysis_results.complexity_score:.2f}")
            
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
                st.success("Optimized Query:")
                formatted_query = format_sql(st.session_state.analysis_results.optimized_query)
                st.code(formatted_query, language="sql")
                if st.button("Copy Optimized Query"):
                    st.session_state.clipboard = formatted_query
                    st.success("Query copied to clipboard!")


def render_manual_analysis_view(analyzer: Optional[QueryAnalyzer]):
    """Render the manual query analysis view.

    Args:
        analyzer: QueryAnalyzer instance
    """
    st.header("Manual Query Analysis")
    
    # Query input methods
    input_method = st.radio(
        "Choose input method",
        ["Direct Input", "File Upload", "Batch Analysis"]
    )
    
    if input_method == "Direct Input":
        st.markdown("### Enter SQL Query")
        query = st.text_area(
            "SQL Query",
            height=200,
            help="Paste your SQL query here for analysis",
            key="sql_input"
        )
        
        # Format button
        if query and st.button("Format Query"):
            formatted_query = format_sql(query)
            st.session_state.sql_input = formatted_query
            st.experimental_rerun()
        
        # Display formatted query with syntax highlighting
        if query:
            st.markdown("### Formatted Query")
            st.code(format_sql(query), language="sql")
        
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
            except json.JSONDecodeError:
                st.error("Invalid JSON format for columns")
                st.session_state.schema_info = None
        
        if st.button("Analyze"):
            if query and analyzer:
                with st.spinner("Analyzing query..."):
                    st.session_state.analysis_results = analyzer.analyze_query(
                        query,
                        schema_info=st.session_state.schema_info
                    )
                    
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader("Upload SQL file", type=["sql"])
        if uploaded_file and analyzer:
            query = uploaded_file.getvalue().decode()
            st.markdown("### SQL Query")
            st.code(format_sql(query), language="sql")
            
            if st.button("Analyze"):
                with st.spinner("Analyzing query..."):
                    st.session_state.analysis_results = analyzer.analyze_query(query)
                    
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
                
                for i, file in enumerate(uploaded_files):
                    query = file.getvalue().decode()
                    st.markdown(f"### Query from {file.name}")
                    st.code(format_sql(query), language="sql")
                    analysis = analyzer.analyze_query(query)
                    results.append({
                        "filename": file.name,
                        "analysis": analysis
                    })
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.session_state.manual_queries = results
    
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
            st.success("Optimized Query:")
            formatted_query = format_sql(st.session_state.analysis_results.optimized_query)
            st.code(formatted_query, language="sql")
            if st.button("Copy Optimized Query"):
                st.session_state.clipboard = formatted_query
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
                    st.code(result['analysis'].optimized_query, language="sql")


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
                    with st.expander("Optimized Query", expanded=True):
                        formatted_query = format_sql(analysis_results.optimized_query)
                        st.code(formatted_query, language="sql")
                        if st.button("Copy to Clipboard"):
                            st.session_state.clipboard = formatted_query
                            st.success("Query copied to clipboard!")


def main():
    """Main function to run the Streamlit application."""
    st.title("Snowflake Query Optimizer")
    st.write("Analyze and optimize your Snowflake SQL queries")

    # Initialize connections
    collector, analyzer = initialize_connections()

    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["Query History Analysis", "Manual Analysis", "Advanced Optimization"]
    )

    if mode == "Query History Analysis":
        render_query_history_view(collector, analyzer)
    elif mode == "Manual Analysis":
        render_manual_analysis_view(analyzer)
    else:
        render_advanced_optimization_view(analyzer)


if __name__ == "__main__":
    main() 