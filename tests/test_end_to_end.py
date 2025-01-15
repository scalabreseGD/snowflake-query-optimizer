import pytest
import streamlit as st
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
from io import BytesIO, StringIO
import json

from snowflake_optimizer.app import (
    initialize_connections,
    render_manual_analysis_view,
    render_query_history_view,
    analyze_query_batch,
    create_excel_report,
    format_sql,
    split_sql_queries,
    analyze_query_callback
)
from snowflake_optimizer.query_analyzer import QueryAnalyzer, QueryAnalysis, SchemaInfo

@pytest.fixture
def mock_secrets():
    """Mock Streamlit secrets."""
    with patch.dict(st.secrets, {
        "SNOWFLAKE_ACCOUNT": "test_account",
        "SNOWFLAKE_USER": "test_user",
        "SNOWFLAKE_PASSWORD": "test_password",
        "SNOWFLAKE_WAREHOUSE": "test_warehouse",
        "ANTHROPIC_API_KEY": "test_api_key"
    }):
        yield

@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state."""
    with patch.object(st, 'session_state', create=True) as mock_state:
        mock_state.formatted_query = ""
        mock_state.analysis_results = None
        mock_state.selected_query = None
        mock_state.batch_results = []
        mock_state.schema_info = None
        mock_state.query_history = None
        yield mock_state

@pytest.fixture
def mock_analyzer():
    """Mock QueryAnalyzer with predefined responses."""
    analyzer = MagicMock(spec=QueryAnalyzer)
    analyzer.analyze_query.return_value = QueryAnalysis(
        original_query="SELECT * FROM test",
        category="Performance",
        complexity_score=0.5,
        confidence_score=0.8,
        antipatterns=["Full table scan"],
        suggestions=["Add WHERE clause"],
        optimized_query="SELECT * FROM test WHERE id > 0"
    )
    return analyzer

def test_end_to_end_manual_analysis(mock_analyzer, mock_session_state):
    """Test end-to-end flow for manual query analysis."""
    # Mock file upload
    test_query = "SELECT * FROM test_table;"
    mock_file = MagicMock()
    mock_file.name = "test.sql"
    mock_file.getvalue.return_value = test_query.encode()
    
    with patch('streamlit.file_uploader') as mock_uploader, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.code') as mock_code, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.radio') as mock_radio, \
         patch('streamlit.markdown') as mock_markdown:
        
        # Setup mocks
        mock_uploader.return_value = mock_file
        mock_button.return_value = True
        mock_radio.return_value = "File Upload"
        
        # Set up session state
        st.session_state.formatted_query = format_sql(test_query)
        st.session_state.analysis_results = None
        st.session_state.selected_query = None
        
        # Render the view
        render_manual_analysis_view(mock_analyzer)
        
        # Simulate analyze button click
        analyze_query_callback(mock_analyzer)
        
        # Verify analysis was performed
        mock_analyzer.analyze_query.assert_called_once_with(
            format_sql(test_query),
            schema_info=None
        )
        assert st.session_state.analysis_results is not None
        assert st.session_state.analysis_results.optimized_query == "SELECT * FROM test WHERE id > 0"
        
        # Verify results were displayed
        mock_code.assert_called()
        mock_markdown.assert_called()

def test_end_to_end_batch_analysis(mock_analyzer, mock_session_state):
    """Test end-to-end flow for batch query analysis."""
    # Mock multiple file uploads
    test_queries = [
        "SELECT * FROM table1;",
        "SELECT id, name FROM table2 WHERE age > 25;"
    ]
    
    mock_files = []
    for i, query in enumerate(test_queries):
        mock_file = MagicMock()
        mock_file.name = f"test{i+1}.sql"
        mock_file.getvalue.return_value = query.encode()
        mock_files.append(mock_file)
    
    with patch('streamlit.file_uploader') as mock_uploader, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.progress') as mock_progress, \
         patch('streamlit.empty') as mock_empty:
        
        # Setup mocks
        mock_uploader.return_value = mock_files
        mock_button.return_value = True
        
        # Process batch
        results = analyze_query_batch(
            [{'filename': f.name, 'query': f.getvalue().decode()} for f in mock_files],
            mock_analyzer
        )
        
        # Verify all queries were analyzed
        assert len(results) == len(test_queries)
        assert mock_analyzer.analyze_query.call_count == len(test_queries)
        
        # Create and verify Excel report
        excel_data = create_excel_report(results)
        assert isinstance(excel_data, bytes)
        
        # Verify Excel content
        with BytesIO(excel_data) as bio:
            df_dict = pd.read_excel(bio, sheet_name=None)
            assert set(df_dict.keys()) == {'Errors', 'Metrics', 'Optimizations'}
            assert len(df_dict['Optimizations']) == len(test_queries)

def test_end_to_end_query_history(mock_analyzer, mock_session_state):
    """Test end-to-end flow for query history analysis."""
    # Mock query history data
    mock_history = pd.DataFrame({
        'QUERY_ID': ['Q1', 'Q2'],
        'QUERY_TEXT': [
            'SELECT * FROM table1',
            'SELECT id, name FROM table2'
        ],
        'EXECUTION_TIME_SECONDS': [10, 20],
        'MB_SCANNED': [100, 200],
        'ROWS_PRODUCED': [1000, 2000]
    })
    
    st.session_state.query_history = mock_history
    
    with patch('streamlit.selectbox') as mock_selectbox, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.dataframe') as mock_dataframe, \
         patch('streamlit.code') as mock_code:
        
        # Setup mocks
        mock_selectbox.return_value = 'Q1'
        mock_button.return_value = True
        
        # Render the view
        render_query_history_view(None, mock_analyzer)
        
        # Verify query was selected and analyzed
        assert st.session_state.selected_query == format_sql(mock_history.iloc[0]['QUERY_TEXT'])
        mock_analyzer.analyze_query.assert_called_once()
        
        # Verify results were displayed
        mock_dataframe.assert_called()
        mock_code.assert_called()

def test_end_to_end_schema_info(mock_analyzer, mock_session_state):
    """Test end-to-end flow with schema information."""
    # Mock schema input
    schema_info = {
        "table_name": "test_table",
        "columns": [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "VARCHAR"}
        ],
        "row_count": 1000
    }
    
    with patch('streamlit.checkbox') as mock_checkbox, \
         patch('streamlit.text_input') as mock_text_input, \
         patch('streamlit.number_input') as mock_number_input, \
         patch('streamlit.text_area') as mock_text_area, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.radio') as mock_radio, \
         patch('streamlit.markdown') as mock_markdown:
        
        # Setup mocks
        mock_checkbox.return_value = True
        mock_text_input.return_value = schema_info["table_name"]
        mock_number_input.return_value = schema_info["row_count"]
        mock_text_area.side_effect = [
            "SELECT * FROM test_table",  # First call for SQL input
            json.dumps(schema_info["columns"])  # Second call for schema columns
        ]
        mock_button.return_value = True
        mock_radio.return_value = "Direct Input"
        
        # Set up session state
        st.session_state.sql_input = "SELECT * FROM test_table"
        st.session_state.formatted_query = "SELECT * FROM test_table"
        st.session_state.selected_query = "SELECT * FROM test_table"
        st.session_state.analysis_results = None
        st.session_state.schema_info = SchemaInfo(
            table_name=schema_info["table_name"],
            columns=schema_info["columns"],
            row_count=schema_info["row_count"]
        )
        
        print("\nDebug - Before render:")
        print(f"sql_input: {st.session_state.sql_input}")
        print(f"formatted_query: {st.session_state.formatted_query}")
        print(f"selected_query: {st.session_state.selected_query}")
        print(f"schema_info: {st.session_state.schema_info}")
        
        # Render the view
        render_manual_analysis_view(mock_analyzer)
        
        print("\nDebug - After render:")
        print(f"sql_input: {st.session_state.sql_input}")
        print(f"formatted_query: {st.session_state.formatted_query}")
        print(f"selected_query: {st.session_state.selected_query}")
        print(f"schema_info: {st.session_state.schema_info}")
        
        # Simulate text area change callback
        if hasattr(mock_text_area, 'on_change'):
            mock_text_area.on_change()
        
        print("\nDebug - Before analyze:")
        print(f"sql_input: {st.session_state.sql_input}")
        print(f"formatted_query: {st.session_state.formatted_query}")
        print(f"selected_query: {st.session_state.selected_query}")
        print(f"schema_info: {st.session_state.schema_info}")
        
        # Simulate analyze button click
        analyze_query_callback(mock_analyzer)
        
        print("\nDebug - After analyze:")
        print(f"sql_input: {st.session_state.sql_input}")
        print(f"formatted_query: {st.session_state.formatted_query}")
        print(f"selected_query: {st.session_state.selected_query}")
        print(f"schema_info: {st.session_state.schema_info}")
        print(f"analysis_results: {st.session_state.analysis_results}")
        
        # Verify analysis used schema info
        mock_analyzer.analyze_query.assert_called_once_with(
            "SELECT * FROM test_table",
            schema_info=st.session_state.schema_info
        )
        assert st.session_state.analysis_results is not None
        assert st.session_state.analysis_results.optimized_query == "SELECT * FROM test WHERE id > 0" 