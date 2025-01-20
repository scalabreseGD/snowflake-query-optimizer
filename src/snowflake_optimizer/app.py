"""Main Streamlit application for Snowflake Query Optimizer."""

import difflib
import io
import json
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import sqlparse
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from trulens.core import TruSession
from trulens.providers.openai import OpenAI as fOpenAI

from snowflake_optimizer.data_collector import QueryMetricsCollector
from snowflake_optimizer.query_analyzer import QueryAnalyzer, SchemaInfo
from snowflake_optimizer.trulens_int import ChatApp, get_trulens_app


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


def setup_dashboard(llm_client: OpenAI, llm_model: str = None,
                    evaluator_llm_client: fOpenAI = None, evaluator_model: str = None):
    if 'tru_session' not in st.session_state:
        session = TruSession()
        session.start_dashboard(port=8502, force=True)
        session.reset_database()
        st.session_state['tru_session'] = session

    if 'tru_chat' not in st.session_state:
        chat = ChatApp(llm_client=llm_client, llm_model=llm_model,
                       evaluator_llm_client=evaluator_llm_client, evaluator_model=evaluator_model)
        st.session_state['tru_chat'] = chat
        tru_app = get_trulens_app(chat, chat.get_feedbacks())
        st.session_state['tru_app'] = tru_app


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

# Define SQL antipatterns with detailed categorization
SQL_ANTIPATTERNS = {
    'PERFORMANCE': {
        'FTS001': {
            'name': 'Full Table Scan',
            'description': 'Query performs a full table scan without appropriate filtering',
            'impact': 'High',
            'detection': ['SELECT *', 'WHERE 1=1', 'no WHERE clause']
        },
        'IJN001': {
            'name': 'Inefficient Join',
            'description': 'Join conditions missing or using non-indexed columns',
            'impact': 'High',
            'detection': ['CROSS JOIN', 'cartesian product', 'missing JOIN condition']
        },
        'IDX001': {
            'name': 'Missing Index',
            'description': 'Frequently filtered or joined columns lack appropriate indexes',
            'impact': 'High',
            'detection': ['frequently filtered column', 'join key without index']
        },
        'LDT001': {
            'name': 'Large Data Transfer',
            'description': 'Query retrieves excessive data volume',
            'impact': 'High',
            'detection': ['SELECT *', 'large table without limit']
        }
    },
    'DATA_QUALITY': {
        'NUL001': {
            'name': 'Unsafe Null Handling',
            'description': 'Improper handling of NULL values in comparisons',
            'impact': 'Medium',
            'detection': ['IS NULL', 'IS NOT NULL', 'NULL comparison']
        },
        'DTM001': {
            'name': 'Data Type Mismatch',
            'description': 'Implicit data type conversions in comparisons',
            'impact': 'Medium',
            'detection': ['implicit conversion', 'type mismatch']
        }
    },
    'COMPLEXITY': {
        'NSQ001': {
            'name': 'Nested Subquery',
            'description': 'Deeply nested subqueries that could be simplified',
            'impact': 'Medium',
            'detection': ['multiple SELECT levels', 'nested subquery']
        },
        'CJN001': {
            'name': 'Complex Join Chain',
            'description': 'Long chain of joins that could be simplified',
            'impact': 'Medium',
            'detection': ['multiple joins', 'join chain']
        }
    },
    'BEST_PRACTICE': {
        'WCD001': {
            'name': 'Wildcard Column Usage',
            'description': 'Using SELECT * instead of specific columns',
            'impact': 'Low',
            'detection': ['SELECT *']
        },
        'ALS001': {
            'name': 'Missing Table Alias',
            'description': 'Tables or subqueries without clear aliases',
            'impact': 'Low',
            'detection': ['missing AS keyword', 'no table alias']
        }
    },
    'SECURITY': {
        'INJ001': {
            'name': 'SQL Injection Risk',
            'description': 'Potential SQL injection vulnerabilities',
            'impact': 'High',
            'detection': ['dynamic SQL', 'string concatenation']
        },
        'PRM001': {
            'name': 'Missing Parameterization',
            'description': 'Hard-coded values instead of parameters',
            'impact': 'Medium',
            'detection': ['literal values', 'hard-coded constants']
        }
    },
    'MAINTAINABILITY': {
        'CMT001': {
            'name': 'Missing Comments',
            'description': 'Complex logic without explanatory comments',
            'impact': 'Low',
            'detection': ['complex logic', 'no comments']
        },
        'FMT001': {
            'name': 'Poor Formatting',
            'description': 'Inconsistent or poor SQL formatting',
            'impact': 'Low',
            'detection': ['inconsistent indentation', 'poor formatting']
        }
    }
}


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
        api_key = st.secrets['API_KEY']
        api_version = st.secrets['API_VERSION']
        api_endpoint = st.secrets['API_ENDPOINT']
        model_name = deployment_name = st.secrets['DEPLOYMENT_NAME']
        azure_openai_client = AzureOpenAI(azure_endpoint=api_endpoint,
                                          api_key=api_key,
                                          api_version=api_version,
                                          )

        api_eval_key = st.secrets['API_EVAL_KEY']
        api_eval_base_url = st.secrets['API_EVAL_BASE_URL']
        eval_model_name = st.secrets['API_EVAL_MODEL_NAME']
        eval_openai_client = fOpenAI(base_url=api_eval_base_url, api_key=api_eval_key, model_engine=eval_model_name)

        # Start Trulens Dashboard
        setup_dashboard(azure_openai_client, model_name, eval_openai_client, eval_model_name)
        logging.debug("Initializing Query Analyzer")
        logging.debug(f"API key length: {len(api_key)}")
        if 'analyzer' not in st.session_state:
            analyzer = QueryAnalyzer(
                openai_client=azure_openai_client,
                openai_model=model_name,
                tru_chat=st.session_state['tru_chat'],
                tru_app=st.session_state['tru_app']
            )
            st.session_state['analyzer'] = analyzer
        else:
            analyzer = st.session_state.analyzer
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
    if not original or not optimized:
        print("Missing query for comparison!")
        return

    st.markdown("### Query Comparison")

    # Create columns for side-by-side view
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Query**")
        formatted_original = format_sql(original)
        st.code(formatted_original, language="sql")

    with col2:
        st.markdown("**Optimized Query**")
        formatted_optimized = format_sql(optimized)
        st.code(formatted_optimized, language="sql")

    # Show diff below
    # st.markdown("### Changes")
    #
    # st.markdown("""
    # <style>
    # .diff-legend {
    #     display: flex;
    #     gap: 20px;
    #     margin-bottom: 10px;
    #     font-family: monospace;
    # }
    # .diff-legend span {
    #     display: inline-flex;
    #     align-items: center;
    #     gap: 5px;
    # }
    # .diff-added { background-color: #2ea04326; }
    # .diff-removed { background-color: #f8514926; }
    # </style>
    # <div class="diff-legend">
    #     <span><span style="color: #2ea043">+</span> Added</span>
    #     <span><span style="color: #f85149">-</span> Removed</span>
    # </div>
    # """, unsafe_allow_html=True)
    #
    # try:
    #     diff_html = create_query_diff(original, optimized)
    #     st.markdown(diff_html, unsafe_allow_html=True)
    # except Exception as e:
    #     print(f"Failed to create or display diff: {str(e)}")
    #     st.error("Failed to display query differences")


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
            value=1,
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
            value=50,
            help="Maximum number of queries to analyze"
        )

        page_size = st.number_input(
            "Page Size",
            min_value=1,
            max_value=1000,
            value=10,
            help="Page Size for Pagination."
        )

        logging.debug(
            f"Query history parameters - days: {days}, min_execution_time: {min_execution_time}, limit: {limit}")

        fetch_query_btn = st.button("Fetch Queries")

        if fetch_query_btn:
            if collector:
                logging.info("Fetching query history from Snowflake")
                with st.spinner("Fetching query history..."):
                    try:
                        if 'current_page' not in st.session_state:
                            st.session_state.current_page = 0

                        # logging.info(f"Successfully fetched {len(st.session_state.query_history)} queries")
                    except Exception as e:
                        logging.error(f"Failed to fetch query history: {str(e)}")
                        st.error(f"Failed to fetch queries: {str(e)}")
            else:
                logging.error("Cannot fetch queries - Snowflake connection not available")
                st.error("Snowflake connection not available")

    with st.container():
        if 'current_page' in st.session_state:
            query_history, total_pages = collector.get_expensive_queries_paginated(
                days=days,
                min_execution_time=min_execution_time,
                limit=limit,
                page_size=page_size,
                page=st.session_state.current_page
            )
            st.info("Select a query from the dataframe below clicking on the left of the table")
            row = st.dataframe(
                query_history[[
                    "query_id",
                    "execution_time_seconds",
                    "mb_scanned",
                    "rows_produced"
                ]],
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )
            prev_col, space_col, next_col = st.columns([1, 3, 1])
            if prev_col.button("Previous") and st.session_state['current_page'] > 0:
                st.session_state['current_page'] -= 1
            if next_col.button("Next") and st.session_state['current_page'] < total_pages - 1:
                st.session_state['current_page'] += 1
            if len(row['selection']['rows']) > 0:
                selected_item = row['selection']['rows'][0]
                selected_query = query_history.iloc[selected_item]
                st.session_state.selected_query = format_sql(selected_query['query_text'])
                st.session_state.formatted_query = st.session_state.selected_query

    with st.container():
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
                st.success("Query Optimization Results")
                st.session_state.formatted_query = format_sql(st.session_state.selected_query)
                display_query_comparison(
                    st.session_state.formatted_query,
                    st.session_state.analysis_results.optimized_query
                )
                if st.button("Copy Optimized Query"):
                    st.session_state.clipboard = format_sql(st.session_state.analysis_results.optimized_query)
                    logging.debug("Optimized query copied to clipboard")
                    st.success("Query copied to clipboard!")


def create_excel_report(batch_results: List[Dict]) -> bytes:
    """Create Excel report from batch analysis results."""
    if not batch_results:
        raise ValueError("No results to export")

    # Prepare data for each sheet
    error_data = []
    metric_data = []
    optimization_data = []

    for result in batch_results:
        analysis = result['analysis']

        # Add error patterns with detailed categorization
        if analysis.antipatterns:
            for pattern in analysis.antipatterns:
                # Find matching antipattern code
                pattern_code = None
                pattern_details = None

                for category, patterns in SQL_ANTIPATTERNS.items():
                    for code, details in patterns.items():
                        if any(detect.lower() in pattern.lower() for detect in details['detection']):
                            pattern_code = code
                            pattern_details = details
                            break
                    if pattern_code:
                        break

                if pattern_code:
                    error_data.append({
                        'Query': result['filename'],
                        'Pattern Code': pattern_code,
                        'Pattern Name': pattern_details['name'],
                        'Category': category,
                        'Description': pattern_details['description'],
                        'Impact': pattern_details['impact'],
                        'Details': pattern,
                        'Suggestion': analysis.suggestions[0] if analysis.suggestions else 'None'
                    })
                else:
                    # Fallback for unrecognized patterns
                    error_data.append({
                        'Query': result['filename'],
                        'Pattern Code': 'UNK001',
                        'Pattern Name': 'Unknown Pattern',
                        'Category': 'Other',
                        'Description': pattern,
                        'Impact': 'Unknown',
                        'Details': pattern,
                        'Suggestion': analysis.suggestions[0] if analysis.suggestions else 'None'
                    })

        # Add metrics data
        metric_data.append({
            'Query': result['filename'],
            'Category': analysis.category,
            'Complexity': analysis.complexity_score,
            'Confidence': analysis.confidence_score
        })

        # Add optimization data
        optimization_data.append({
            'Query': result['filename'],
            'Original': result['original_query'],
            'Optimized': analysis.optimized_query if analysis.optimized_query else 'No optimization needed',
            'Suggestions': '\n'.join(analysis.suggestions) if analysis.suggestions else 'None'
        })

    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write error patterns sheet
        if error_data:
            pd.DataFrame(error_data).to_excel(writer, sheet_name='Errors', index=False)
        else:
            pd.DataFrame(
                columns=['Query', 'Pattern Code', 'Pattern Name', 'Category', 'Description', 'Impact', 'Details',
                         'Suggestion']).to_excel(writer, sheet_name='Errors', index=False)

        # Write metrics sheet
        if metric_data:
            pd.DataFrame(metric_data).to_excel(writer, sheet_name='Metrics', index=False)
        else:
            pd.DataFrame(columns=['Query', 'Category', 'Complexity', 'Confidence']).to_excel(writer,
                                                                                             sheet_name='Metrics',
                                                                                             index=False)

        # Write optimizations sheet
        if optimization_data:
            pd.DataFrame(optimization_data).to_excel(writer, sheet_name='Optimizations', index=False)
        else:
            pd.DataFrame(columns=['Query', 'Original', 'Optimized', 'Suggestions']).to_excel(writer,
                                                                                             sheet_name='Optimizations',
                                                                                             index=False)

        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column[0].column_letter].width = min(adjusted_width,
                                                                                 100)  # Cap width at 100

    # Get the bytes value
    excel_data = output.getvalue()
    output.close()

    return excel_data


def analyze_query_callback(analyzer: Optional[QueryAnalyzer]):
    """Callback function for analyzing queries in the manual analysis view.
    
    Args:
        analyzer: QueryAnalyzer instance to use for analysis
    """
    print("\n=== Starting Query Analysis Callback ===")

    if not analyzer:
        print("No analyzer available")
        st.error("Query analyzer is not initialized")
        return

    if not st.session_state.formatted_query:
        print("No query to analyze")
        return

    try:
        print(f"Analyzing query of length: {len(st.session_state.formatted_query)}")
        st.session_state.analysis_results = analyzer.analyze_query(
            st.session_state.formatted_query,
            schema_info=st.session_state.schema_info if hasattr(st.session_state, 'schema_info') else None
        )
        print("Analysis completed successfully")

        # Store the analyzed query for comparison
        st.session_state.selected_query = st.session_state.formatted_query

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        st.error(f"Analysis failed: {str(e)}")
        st.session_state.analysis_results = None


def render_manual_analysis_view(analyzer: Optional[QueryAnalyzer]):
    """Render the manual query analysis view."""
    print("\n=== Starting Manual Analysis View ===")
    st.header("Manual Query Analysis")

    if not analyzer:
        st.error("Query analyzer is not initialized. Please check your configuration.")
        return

    # Initialize session state variables if they don't exist
    if "formatted_query" not in st.session_state:
        st.session_state.formatted_query = ""
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "selected_query" not in st.session_state:
        st.session_state.selected_query = None
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []

    # Query input methods
    input_method = st.radio(
        "Choose input method",
        ["Direct Input", "File Upload", "Batch Analysis"]
    )

    # Store current batch results
    current_batch_results = st.session_state.batch_results.copy() if hasattr(st.session_state, 'batch_results') else []

    if input_method == "Direct Input":
        st.markdown("### Enter SQL Query")

        # Format button before text area
        if st.button("Format Query"):
            if st.session_state.formatted_query:
                st.session_state.formatted_query = format_sql(st.session_state.formatted_query)

        # Text area for SQL input
        query = st.text_area(
            "SQL Query",
            value=st.session_state.formatted_query,
            height=200,
            help="Paste your SQL query here for analysis",
            key="sql_input"
        )

        # Update formatted_query only if query has changed
        if query != st.session_state.formatted_query:
            st.session_state.formatted_query = query

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
            except json.JSONDecodeError:
                st.error("Invalid JSON format for columns")
                st.session_state.schema_info = None

        analyze_button = st.button("Analyze", on_click=lambda: analyze_query_callback(analyzer))

    elif input_method == "File Upload":
        st.markdown("### Upload SQL File")
        uploaded_file = st.file_uploader("Choose a SQL file", type=["sql"])

        if uploaded_file:
            query = uploaded_file.getvalue().decode()
            st.session_state.formatted_query = format_sql(query)
            st.markdown("### Preview")
            st.code(st.session_state.formatted_query, language="sql")

            analyze_button = st.button("Analyze", on_click=lambda: analyze_query_callback(analyzer))

    elif input_method == "Batch Analysis":
        st.markdown("### Upload SQL Files")
        uploaded_files = st.file_uploader("Choose SQL files", type=["sql"], accept_multiple_files=True)

        if uploaded_files:
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process uploaded files
            all_queries = []
            for sql_file in uploaded_files:
                content = sql_file.getvalue().decode()
                queries = split_sql_queries(content)
                print(f"\n=== Splitting SQL Queries ===")
                print(f"Found {len(queries)} queries in {sql_file.name}")
                for i, query in enumerate(queries):
                    print(f"Found valid query of length: {len(query)}")
                    all_queries.append({
                        'filename': f"{sql_file.name} (Query {i + 1})",
                        'query': query
                    })
            print(f"Total queries found: {len(all_queries)}")

            analyze_button = st.button("Analyze All", key="batch_analyze")
            if analyze_button:
                results = []
                for i, query_info in enumerate(all_queries):
                    progress = (i + 1) / len(all_queries)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing {query_info['filename']}...")

                    try:
                        formatted_query = format_sql(query_info['query'])
                        analysis = analyzer.analyze_query(formatted_query)
                        if analysis:
                            results.append({
                                'filename': query_info['filename'],
                                'original_query': query_info['query'],
                                'analysis': analysis
                            })
                    except Exception as e:
                        st.error(f"Failed to analyze {query_info['filename']}: {str(e)}")

                st.session_state.batch_results = results
                status_text.text("Analysis complete!")

            # Display results if available
            if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results:
                st.markdown("### Analysis Results")
                for result in st.session_state.batch_results:
                    with st.expander(f"Results for {result['filename']}"):
                        st.code(result['original_query'], language="sql")
                        st.info(f"Category: {result['analysis'].category}")
                        st.progress(result['analysis'].complexity_score,
                                    text=f"Complexity Score: {result['analysis'].complexity_score:.2f}")

                        if result['analysis'].antipatterns:
                            st.warning("Antipatterns:")
                            for pattern in result['analysis'].antipatterns:
                                st.write(f"- {pattern}")

                        if result['analysis'].suggestions:
                            st.info("Suggestions:")
                            for suggestion in result['analysis'].suggestions:
                                st.write(f"- {suggestion}")

                        if result['analysis'].optimized_query:
                            st.success("Optimized Query:")
                            st.code(result['analysis'].optimized_query, language="sql")

            # Export functionality in a separate section
            if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results:
                st.markdown("### Export Results")
                try:
                    print("\n=== Export Button Clicked ===")
                    print(f"Creating Excel report for {len(st.session_state.batch_results)} results")
                    excel_data = create_excel_report(st.session_state.batch_results)
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_data,
                        file_name="query_analysis_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel"  # Add unique key
                    )
                except Exception as e:
                    st.error(f"Failed to create Excel report: {str(e)}")
                    print(f"Excel export error: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")

    # Display analysis results if available
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
            st.success("Query Optimization Results")
            display_query_comparison(
                st.session_state.formatted_query,
                st.session_state.analysis_results.optimized_query
            )
            if st.button("Copy Optimized Query"):
                st.session_state.clipboard = format_sql(st.session_state.analysis_results.optimized_query)
                st.success("Query copied to clipboard!")


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


def split_sql_queries(content: str) -> List[str]:
    """Split SQL content into individual queries based on blank lines.
    
    Args:
        content: String containing multiple SQL queries separated by blank lines
        
    Returns:
        List of individual SQL queries
    """
    print("\n=== Splitting SQL Queries ===")
    # Remove comments and empty lines
    lines = []
    in_multiline_comment = False

    for line in content.splitlines():
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Handle multiline comments
        if line.startswith('/*'):
            in_multiline_comment = True
            continue
        if '*/' in line:
            in_multiline_comment = False
            continue
        if in_multiline_comment:
            continue

        # Handle single line comments
        if line.startswith('--'):
            continue

        lines.append(line)

    # Join lines back together
    content = ' '.join(lines)

    # Split by semicolon and filter
    queries = []
    current_query = []

    for line in content.split(';'):
        line = line.strip()
        if line:
            # Basic SQL validation
            if any(keyword in line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER']):
                queries.append(line + ';')
                print(f"Found valid query of length: {len(line)}")
            else:
                print(f"Skipping invalid SQL: {line[:50]}...")

    print(f"Total queries found: {len(queries)}")
    return queries


def analyze_query_with_retry(analyzer: QueryAnalyzer, query: str, schema_info: Optional[SchemaInfo] = None,
                             max_retries: int = 3) -> Optional[Any]:
    """Analyze a query with retry logic and error handling.
    
    Args:
        analyzer: QueryAnalyzer instance
        query: SQL query to analyze
        schema_info: Optional schema information
        max_retries: Maximum number of retry attempts
        
    Returns:
        Analysis results or None if analysis fails
    """
    print(f"\n=== Analyzing Query (length: {len(query)}) ===")

    for attempt in range(max_retries):
        try:
            # Try to format the query first
            formatted_query = format_sql(query)
            print(f"Attempt {attempt + 1}: Formatted query length: {len(formatted_query)}")

            # Analyze the formatted query
            result = analyzer.analyze_query(
                formatted_query,
                schema_info=schema_info
            )
            print("Analysis successful")
            return result

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to analyze query after {max_retries} attempts: {str(e)}")
            continue

    return None


def group_related_queries(queries: List[Dict]) -> List[Dict]:
    """Group related queries based on common tables and patterns.
    
    Args:
        queries: List of query dictionaries with analysis results
        
    Returns:
        List of query groups with optimization suggestions
    """
    groups = []
    processed = set()

    for i, query in enumerate(queries):
        if i in processed:
            continue

        related = {i}
        base_tables = set(query.get('tables', []))

        # Find related queries
        for j, other in enumerate(queries):
            if j != i and j not in processed:
                other_tables = set(other.get('tables', []))
                # If queries share tables or have similar patterns
                if (base_tables & other_tables) or (
                        query['analysis'].category == other['analysis'].category
                ):
                    related.add(j)

        # Create group
        group = {
            'queries': [queries[idx] for idx in related],
            'common_tables': base_tables,
            'category': query['analysis'].category,
            'group_suggestions': []
        }

        # Aggregate group-level suggestions
        all_suggestions = []
        for q in group['queries']:
            if q['analysis'].suggestions:
                all_suggestions.extend(q['analysis'].suggestions)

        # Find common suggestions
        if all_suggestions:
            from collections import Counter
            suggestion_counts = Counter(all_suggestions)
            group['group_suggestions'] = [
                sugg for sugg, count in suggestion_counts.items()
                if count > 1  # Suggestion appears in multiple queries
            ]

        groups.append(group)
        processed.update(related)

    return groups


def get_error_analysis_prompt(query: str) -> str:
    """Generate the prompt for LLM to analyze SQL query errors and anti-patterns.
    
    Args:
        query: SQL query to analyze
        
    Returns:
        Formatted prompt for LLM
    """
    return f"""Analyze the following SQL query for potential errors, anti-patterns, and optimization opportunities. 
Provide a detailed analysis in the following JSON format:

{{
    "error_patterns": [
        {{
            "category": "One of: PERFORMANCE, DATA_QUALITY, COMPLEXITY, BEST_PRACTICE, SECURITY, MAINTAINABILITY",
            "code": "One of: FTS001, IJN001, IDX001, LDT001, NUL001, DTM001, NSQ001, CJN001, WCD001, ALS001, INJ001, PRM001, CMT001, FMT001",
            "pattern": "Name of the anti-pattern",
            "severity": "One of: High, Medium, Low",
            "impact": "Brief impact description",
            "location": "Where in the query this occurs",
            "description": "Detailed explanation of the issue",
            "recommendation": "How to fix or improve"
        }}
    ],
    "query_metrics": {{
        "complexity_score": "Score from 0-1",
        "readability_score": "Score from 0-1",
        "maintainability_score": "Score from 0-1",
        "performance_impact": "Score from 0-1",
        "structural_metrics": {{
            "num_joins": "Number",
            "num_conditions": "Number",
            "nesting_depth": "Number",
            "num_aggregations": "Number"
        }}
    }}
}}

Anti-pattern Categories and Codes:
1. PERFORMANCE:
   - FTS001: Full Table Scan - Query performs a full table scan without appropriate filtering
   - IJN001: Inefficient Join - Join conditions missing or using non-indexed columns
   - IDX001: Missing Index - Frequently filtered or joined columns lack appropriate indexes
   - LDT001: Large Data Transfer - Query retrieves excessive data volume

2. DATA_QUALITY:
   - NUL001: Unsafe Null Handling - Improper handling of NULL values in comparisons
   - DTM001: Data Type Mismatch - Implicit data type conversions in comparisons

3. COMPLEXITY:
   - NSQ001: Nested Subquery - Deeply nested subqueries that could be simplified
   - CJN001: Complex Join Chain - Long chain of joins that could be simplified

4. BEST_PRACTICE:
   - WCD001: Wildcard Column Usage - Using SELECT * instead of specific columns
   - ALS001: Missing Table Alias - Tables or subqueries without clear aliases

5. SECURITY:
   - INJ001: SQL Injection Risk - Potential SQL injection vulnerabilities
   - PRM001: Missing Parameterization - Hard-coded values instead of parameters

6. MAINTAINABILITY:
   - CMT001: Missing Comments - Complex logic without explanatory comments
   - FMT001: Poor Formatting - Inconsistent or poor SQL formatting

Focus on:
1. Performance implications
2. Logical correctness
3. Best practices
4. Maintainability issues
5. Security concerns
6. Complexity assessment

Query to analyze:
{query}

Provide the analysis in the exact JSON format specified above."""


def analyze_query_batch(queries: List[Dict], analyzer: QueryAnalyzer, schema_info: Optional[SchemaInfo] = None) -> List[
    Dict]:
    """Analyze a batch of queries in parallel using multi-threading.
    
    Args:
        queries: List of query dictionaries containing filename and query
        analyzer: QueryAnalyzer instance
        schema_info: Optional schema information
        
    Returns:
        List of analysis results
    """
    print("\n=== Starting Batch Analysis ===")
    print(f"Number of queries to analyze: {len(queries)}")
    results = []

    # Calculate optimal number of workers
    max_workers = min(32, len(queries))  # Cap at 32 threads
    print(f"Using {max_workers} worker threads")

    def analyze_single_query(query_info: Dict) -> Optional[Dict]:
        try:
            print(f"\nAnalyzing query from {query_info['filename']}")
            analysis_result = analyzer.analyze_query(
                query_info['query'],
                schema_info=schema_info
            )

            if analysis_result:
                print(f"Analysis successful for {query_info['filename']}")
                return {
                    "filename": query_info['filename'],
                    "original_query": query_info['query'],
                    "analysis": analysis_result
                }
            print(f"No analysis result for {query_info['filename']}")
            return None

        except Exception as e:
            print(f"Error analyzing {query_info['filename']}: {str(e)}")
            return None

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and get futures
        future_to_query = {
            executor.submit(analyze_single_query, query_info): query_info
            for query_info in queries
        }

        # Process completed futures as they finish
        for future in as_completed(future_to_query):
            query_info = future_to_query[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"Added result for {query_info['filename']} to results list")
            except Exception as e:
                print(f"Analysis failed for {query_info['filename']}: {str(e)}")

    print(f"\nBatch analysis completed. Total results: {len(results)}")
    return results

def compare_query(collector):

    query_id = '01b9d4ef-0a08-cfb7-0000-5f21c984fa8e'
    optimized = """
    SELECT count(*) AS cnt,
       'CRM' AS env
FROM
    (SELECT DISTINCT 'arbys' AS brand_id,
                     'sfmc' AS source_system_name,
                     account_id,
                     job_id,
                     subscriber_key,
                     external_id,
                     customer_id,
                     offer_id,
                     batch_id,
                     list_id,
                     creative_variant,
                     event_date,
                     offer_code,
                     sub_id,
                     triggered_send_id,
                     error_code,
                     datasource_name,
                     email_address,
                     offer_name,
                     parent_offer_id,
                     journey_name,
                     journey_step,
                     user_defined_segment_1,
                     user_defined_segment_2,
                     user_defined_segment_3,
                     user_defined_segment_4,
                     user_defined_segment_5,
                     offer_decision_logic,
                     email_name,
                     campaign_id,
                     to_varchar(mdt_created_on, 'YYYYMMDD')::INTEGER AS load_id,
                     split_part(mdt_filename, '/', -1) AS load_filename
     FROM crm.arbys.sfmc_sendlog)
UNION ALL
SELECT count(*) AS cnt,
       'IDS' AS env
FROM
    (SELECT DISTINCT brand_id,
                     source_system_name,
                     account_id,
                     job_id,
                     subscriber_key,
                     profile_id,
                     external_id,
                     customer_id,
                     offer_id,
                     batch_id,
                     list_id,
                     creative_variant,
                     event_dttm,
                     offer_code,
                     subscriber_id,
                     triggered_send_external_key,
                     error_code,
                     datasource_name,
                     email_address,
                     offer_name,
                     parent_offer_id,
                     journey_name,
                     journey_step,
                     strength_of_customer,
                     user_defined_segment_1,
                     user_defined_segment_2,
                     user_defined_segment_3,
                     user_defined_segment_4,
                     user_defined_segment_5,
                     offer_decision_logic,
                     offer_1_decision_logic,
                     point_balance,
                     email_name,
                     campaign_id,
                     campaign_name,
                     campaign_type,
                     campaign_category_type,
                     campaign_description,
                     campaign_objective,
                     campaign_start_date,
                     campaign_end_date,
                     campaign_duration,
                     load_filename
     FROM ids_qa.cust.crm_campaign_send_log_arbys);
    """
    collector.compare_optimized_query_with_original(optimized, query_id)


def main():
    """Main function to run the Streamlit application."""
    logging.info("Starting main application")
    st.title("Snowflake Query Optimizer")
    st.write("Analyze and optimize your Snowflake SQL queries")

    # Initialize connections
    collector, analyzer = initialize_connections()
    compare_query(collector)

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
