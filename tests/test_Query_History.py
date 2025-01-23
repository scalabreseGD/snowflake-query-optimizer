import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from snowflake_optimizer.pages.Query_History import (
    render_query_history_view,
    get_databases,
    main
)


@pytest.fixture
def mock_streamlit():
    with patch('snowflake_optimizer.pages.Query_History.st') as mock_st:
        mock_st.session_state = {}
        mock_st.slider.return_value = 1
        mock_st.number_input.return_value = 60
        mock_st.selectbox.return_value = ''
        mock_st.cache_data = lambda show_spinner: lambda func: func
        yield mock_st


@pytest.fixture
def mock_collector():
    collector = Mock()
    collector.get_databases.return_value = [{'database_name': 'db1'}, {'database_name': 'db2'}]
    collector.get_expensive_queries_paginated.return_value = (
        pd.DataFrame({
            'query_id': ['id1', 'id2'],
            'execution_time_seconds': [100, 200],
            'mb_scanned': [1000, 2000],
            'rows_produced': [500, 1000],
            'query_text': ['SELECT * FROM table1', 'SELECT * FROM table2']
        }),
        2  # total_pages
    )
    return collector


@pytest.fixture
def mock_analyzer():
    analyzer = Mock()
    analyzer.analyze_query.return_value = [{
        'filename': 'id1',
        'analysis': 'Test analysis',
        'recommendations': ['rec1', 'rec2']
    }]
    return analyzer


@pytest.fixture
def mock_executor():
    return Mock()


def test_render_query_history_view(mock_streamlit, mock_collector, mock_analyzer, mock_executor):
    with patch('snowflake_optimizer.pages.Query_History.setup_logging'), \
            patch('snowflake_optimizer.pages.Query_History.load_dotenv'), \
            patch('snowflake_optimizer.pages.Query_History.init_common_states'):
        render_query_history_view('test_page', mock_collector, mock_analyzer, mock_executor)

        mock_streamlit.header.assert_called_with("Query History Analysis")
        mock_streamlit.slider.assert_called()
        mock_streamlit.number_input.assert_called()


def test_fetch_queries_success(mock_streamlit, mock_collector, mock_analyzer, mock_executor):
    mock_streamlit.button.return_value = True

    with patch('snowflake_optimizer.pages.Query_History.setup_logging'), \
            patch('snowflake_optimizer.pages.Query_History.load_dotenv'), \
            patch('snowflake_optimizer.pages.Query_History.init_common_states'):
        render_query_history_view('test_page', mock_collector, mock_analyzer, mock_executor)

        assert 'test_page_current_page' in mock_streamlit.session_state
        mock_collector.get_expensive_queries_paginated.assert_called()


def test_fetch_queries_no_collector(mock_streamlit, mock_analyzer, mock_executor):
    mock_streamlit.button.return_value = True

    with patch('snowflake_optimizer.pages.Query_History.setup_logging'), \
            patch('snowflake_optimizer.pages.Query_History.load_dotenv'), \
            patch('snowflake_optimizer.pages.Query_History.init_common_states'):
        render_query_history_view('test_page', None, mock_analyzer, mock_executor)

        mock_streamlit.error.assert_called_with("Snowflake connection not available")


def test_query_analysis_flow(mock_streamlit, mock_collector, mock_analyzer, mock_executor):
    mock_streamlit.session_state = {
        'test_page_selected_query': 'SELECT * FROM table1',
        'test_page_formatted_query': 'SELECT * FROM table1'
    }
    mock_streamlit.dataframe.return_value = {'selection': {'rows': [0]}}

    with patch('snowflake_optimizer.pages.Query_History.setup_logging'), \
            patch('snowflake_optimizer.pages.Query_History.load_dotenv'), \
            patch('snowflake_optimizer.pages.Query_History.init_common_states'), \
            patch('snowflake_optimizer.pages.Query_History.format_sql') as mock_format_sql, \
            patch('snowflake_optimizer.pages.Query_History.evaluate_or_repair_query') as mock_evaluate:
        mock_format_sql.return_value = 'formatted SQL'
        mock_evaluate.return_value = {'analysis': 'Test analysis'}

        render_query_history_view('test_page', mock_collector, mock_analyzer, mock_executor)

        mock_analyzer.analyze_query.assert_called()
        assert 'test_page_analysis_results' in mock_streamlit.session_state


def test_get_databases(mock_streamlit, mock_collector):
    result = get_databases(mock_collector)
    assert result == ['', 'db1', 'db2']
    mock_collector.get_databases.assert_called_once()


def test_pagination_controls(mock_streamlit, mock_collector, mock_analyzer, mock_executor):
    mock_streamlit.session_state = {'test_page_current_page': 0}
    mock_streamlit.columns.return_value = [Mock(), Mock(), Mock()]

    with patch('snowflake_optimizer.pages.Query_History.setup_logging'), \
            patch('snowflake_optimizer.pages.Query_History.load_dotenv'), \
            patch('snowflake_optimizer.pages.Query_History.init_common_states'):
        render_query_history_view('test_page', mock_collector, mock_analyzer, mock_executor)

        mock_collector.get_expensive_queries_paginated.assert_called_with(
            days=1,
            min_execution_time=60,
            limit=50,
            page_size=10,
            page=0,
            db_schema_filter=''
        )


def test_main():
    with patch('snowflake_optimizer.pages.Query_History.st') as mock_st, \
            patch('snowflake_optimizer.pages.Query_History.initialize_connections') as mock_init, \
            patch('snowflake_optimizer.pages.Query_History.get_snowflake_query_executor') as mock_executor, \
            patch('snowflake_optimizer.pages.Query_History.render_query_history_view') as mock_render:
        mock_init.return_value = (Mock(), Mock())
        mock_executor.return_value = Mock()

        main()

        mock_st.set_page_config.assert_called_with(page_title="Query History")
        mock_init.assert_called()
        mock_executor.assert_called()
        mock_render.assert_called()