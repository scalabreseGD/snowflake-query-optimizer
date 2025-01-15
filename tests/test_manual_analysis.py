import pytest
import streamlit as st
from unittest.mock import MagicMock, patch
from snowflake_optimizer.app import analyze_query_callback, render_manual_analysis_view
from snowflake_optimizer.query_analyzer import QueryAnalyzer, QueryAnalysis

@pytest.fixture
def mock_analyzer():
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

@pytest.fixture
def mock_session_state():
    with patch.object(st, 'session_state', create=True) as mock_state:
        mock_state.formatted_query = "SELECT * FROM test"
        mock_state.analysis_results = None
        mock_state.schema_info = None
        yield mock_state

def test_analyze_query_callback(mock_analyzer, mock_session_state):
    """Test that analyze_query_callback correctly processes queries and updates session state."""
    # Setup
    st.session_state.formatted_query = "SELECT * FROM test"
    
    # Call the callback
    analyze_query_callback(mock_analyzer)
    
    # Verify the analyzer was called with correct parameters
    mock_analyzer.analyze_query.assert_called_once_with(
        "SELECT * FROM test",
        schema_info=None
    )
    
    # Verify session state was updated
    assert st.session_state.analysis_results is not None
    assert st.session_state.analysis_results.category == "Performance"
    assert st.session_state.analysis_results.optimized_query == "SELECT * FROM test WHERE id > 0"

def test_analyze_query_callback_with_schema(mock_analyzer, mock_session_state):
    """Test analyze_query_callback with schema information."""
    # Setup
    st.session_state.formatted_query = "SELECT * FROM test"
    st.session_state.schema_info = {
        "table_name": "test",
        "columns": [{"name": "id", "type": "INTEGER"}],
        "row_count": 1000
    }
    
    # Call the callback
    analyze_query_callback(mock_analyzer)
    
    # Verify analyzer was called with schema info
    mock_analyzer.analyze_query.assert_called_once()
    call_args = mock_analyzer.analyze_query.call_args[1]
    assert call_args["schema_info"] == st.session_state.schema_info

def test_analyze_query_callback_empty_query(mock_analyzer, mock_session_state):
    """Test analyze_query_callback behavior with empty query."""
    # Setup
    st.session_state.formatted_query = ""
    
    # Call the callback
    analyze_query_callback(mock_analyzer)
    
    # Verify analyzer was not called
    mock_analyzer.analyze_query.assert_not_called()
    
    # Verify session state wasn't updated
    assert st.session_state.analysis_results is None

def test_analyze_query_callback_error_handling(mock_analyzer, mock_session_state):
    """Test error handling in analyze_query_callback."""
    # Setup
    st.session_state.formatted_query = "SELECT * FROM test"
    mock_analyzer.analyze_query.side_effect = Exception("Test error")
    
    # Call the callback
    analyze_query_callback(mock_analyzer)
    
    # Verify error handling
    assert st.session_state.analysis_results is None

def test_analyze_query_callback_no_analyzer(mock_session_state):
    """Test analyze_query_callback behavior without analyzer."""
    # Setup
    st.session_state.formatted_query = "SELECT * FROM test"
    
    # Call the callback
    analyze_query_callback(None)
    
    # Verify error handling
    assert st.session_state.analysis_results is None

def test_render_manual_analysis_view(mock_analyzer):
    """Test the manual analysis view rendering."""
    with patch('streamlit.header') as mock_header:
        render_manual_analysis_view(mock_analyzer)
        mock_header.assert_called_once_with("Manual Query Analysis")

def test_render_manual_analysis_view_no_analyzer():
    """Test manual analysis view behavior without analyzer."""
    with patch('streamlit.error') as mock_error:
        render_manual_analysis_view(None)
        mock_error.assert_called_once() 