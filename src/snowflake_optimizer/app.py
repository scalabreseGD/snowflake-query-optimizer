"""Main Streamlit application for Snowflake Query Optimizer."""

import os
from typing import Optional, List, Dict
import json
import streamlit as st
from dotenv import load_dotenv

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
                st.session_state.selected_query = st.session_state.query_history[
                    st.session_state.query_history["QUERY_ID"] == selected_query_id
                ]["QUERY_TEXT"].iloc[0]

    with col2:
        if st.session_state.selected_query:
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
                st.code(
                    st.session_state.analysis_results.optimized_query,
                    language="sql"
                )


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
        query = st.text_area(
            "Enter your SQL query",
            height=200,
            help="Paste your SQL query here for analysis"
        )
        
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
            st.code(query, language="sql")
            
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
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info(f"Query Category: {st.session_state.analysis_results.category}")
        with col2:
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
                
        # Display index suggestions
        if st.session_state.analysis_results.index_suggestions:
            st.info("Index Suggestions:")
            for suggestion in st.session_state.analysis_results.index_suggestions:
                st.write(f"- {suggestion}")

        # Display optimized query
        if st.session_state.analysis_results.optimized_query:
            st.success("Optimized Query:")
            st.code(
                st.session_state.analysis_results.optimized_query,
                language="sql"
            )
    
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


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Snowflake Query Optimizer",
        page_icon="❄️",
        layout="wide"
    )

    st.title("❄️ Snowflake Query Optimizer")
    st.write(
        "Analyze and optimize your Snowflake SQL queries using AI-powered recommendations."
    )

    # Initialize connections
    collector, analyzer = initialize_connections()

    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["Query History Analysis", "Manual Query Analysis"]
    )

    if mode == "Query History Analysis":
        render_query_history_view(collector, analyzer)
    else:
        render_manual_analysis_view(analyzer)


if __name__ == "__main__":
    main() 