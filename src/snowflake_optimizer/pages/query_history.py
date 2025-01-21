import logging

import streamlit as st
from typing import Optional

from dotenv import load_dotenv

from snowflake_optimizer.connections import setup_logging, initialize_connections
from snowflake_optimizer.data_collector import QueryMetricsCollector
from snowflake_optimizer.query_analyzer import QueryAnalyzer
from snowflake_optimizer.utils import format_sql, display_query_comparison


def render_query_history_view(page_id: str, collector: Optional[QueryMetricsCollector],
                              analyzer: Optional[QueryAnalyzer]):
    """Render the query history analysis view.

    Args:
        collector: QueryMetricsCollector instance
        analyzer: QueryAnalyzer instance
    """

    # Set up logging
    setup_logging()

    # Load environment variables
    load_dotenv()

    if f"{page_id}_selected_query" not in st.session_state:
        st.session_state[f"{page_id}_selected_query"] = None
    if f"{page_id}_formatted_query" not in st.session_state:
        st.session_state[f"{page_id}_formatted_query"] = ""
    if f"{page_id}_analysis_results" not in st.session_state:
        st.session_state[f"{page_id}_analysis_results"] = None
    if f"{page_id}_clipboard" not in st.session_state:
        st.session_state[f"{page_id}_clipboard"] = None

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
                        if f"{page_id}_current_page" not in st.session_state:
                            st.session_state[f"{page_id}_current_page"] = 0
                    except Exception as e:
                        logging.error(f"Failed to fetch query history: {str(e)}")
                        st.error(f"Failed to fetch queries: {str(e)}")
            else:
                logging.error("Cannot fetch queries - Snowflake connection not available")
                st.error("Snowflake connection not available")

    with st.container():
        if f"{page_id}_current_page" in st.session_state:
            query_history, total_pages = collector.get_expensive_queries_paginated(
                days=days,
                min_execution_time=min_execution_time,
                limit=limit,
                page_size=page_size,
                page=st.session_state[f"{page_id}_current_page"]
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
                st.session_state[f"{page_id}_selected_query"] = format_sql(selected_query['query_text'])
                st.session_state[f"{page_id}_formatted_query"] = st.session_state[f"{page_id}_selected_query"]

    with st.container():
        if st.session_state[f"{page_id}_selected_query"]:
            st.markdown("### Selected Query")
            st.code(st.session_state[f"{page_id}_selected_query"], language="sql")

            if st.button("Analyze Query"):
                if analyzer:
                    logging.info("Starting query analysis")
                    with st.spinner("Analyzing query..."):
                        try:
                            st.session_state[f"{page_id}_analysis_results"] = analyzer.analyze_query(
                                st.session_state[f"{page_id}_selected_query"]
                            )
                            logging.info("Query analysis completed successfully")
                        except Exception as e:
                            logging.error(f"Query analysis failed: {str(e)}")
                            st.error(f"Analysis failed: {str(e)}")

        if st.session_state[f"{page_id}_analysis_results"]:
            st.subheader("Analysis Results")

            # Log analysis results
            logging.debug(f'Analysis results - Category: {st.session_state[f"{page_id}_analysis_results"].category}, '
                          f'Complexity: {st.session_state[f"{page_id}_analysis_results"].complexity_score:.2f}')

            # Display query category and complexity
            st.info(f'Query Category: {st.session_state[f"{page_id}_analysis_results"].category}')
            st.progress(st.session_state[f"{page_id}_analysis_results"].complexity_score,
                        text=f'Complexity Score: {st.session_state[f"{page_id}_analysis_results"].complexity_score:.2f}')

            # Display antipatterns
            if st.session_state[f"{page_id}_analysis_results"].antipatterns:
                logging.debug(
                    f'Antipatterns detected: {len(st.session_state[f"{page_id}_analysis_results"].antipatterns)}')
                st.warning("Antipatterns Detected:")
                for pattern in st.session_state[f"{page_id}_analysis_results"].antipatterns:
                    st.write(f"- {pattern}")

            # Display suggestions
            if st.session_state[f"{page_id}_analysis_results"].suggestions:
                logging.debug(
                    f'Optimization suggestions: {len(st.session_state[f"{page_id}_analysis_results"].suggestions)}')
                st.info("Optimization Suggestions:")
                for suggestion in st.session_state[f"{page_id}_analysis_results"].suggestions:
                    st.write(f"- {suggestion}")

            # Display optimized query
            if st.session_state[f"{page_id}_analysis_results"].optimized_query:
                logging.info("Optimized query generated")
                st.success("Query Optimization Results")
                st.session_state[f"{page_id}_formatted_query"] = format_sql(
                    st.session_state[f"{page_id}_selected_query"])
                display_query_comparison(
                    st.session_state[f"{page_id}_formatted_query"],
                    st.session_state[f"{page_id}_analysis_results"].optimized_query
                )
                if st.button("Copy Optimized Query"):
                    st.session_state[f"{page_id}_clipboard"] = format_sql(
                        st.session_state[f"{page_id}_analysis_results"].optimized_query)
                    logging.debug("Optimized query copied to clipboard")
                    st.success("Query copied to clipboard!")


def main():
    st.title("Snowflake Query Optimizer")
    st.write("Analyze and optimize your Snowflake SQL queries")
    page_id = 'query_history'
    # Initialize connections
    _collector, _analyzer = initialize_connections(page_id)
    render_query_history_view(page_id, _collector, _analyzer)

main()