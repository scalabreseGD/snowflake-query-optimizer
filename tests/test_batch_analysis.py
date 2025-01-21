import pytest
import streamlit as st
from unittest.mock import MagicMock, patch, mock_open
import json
from io import StringIO
import time

from snowflake_optimizer.query_analyzer import QueryAnalyzer, QueryAnalysis, SchemaInfo, InputAnalysisModel
from snowflake_optimizer.utils import split_sql_queries


@pytest.fixture
def mock_analyzer():
    """Mock QueryAnalyzer with predefined responses."""
    analyzer = MagicMock(spec=QueryAnalyzer)
    analyzer.analyze_query.return_value = [QueryAnalysis(
        original_query="SELECT * FROM test",
        category="Performance",
        complexity_score=0.5,
        confidence_score=0.8,
        antipatterns=["Full table scan"],
        suggestions=["Add WHERE clause"],
        optimized_query="SELECT * FROM test WHERE id > 0"
    )]
    return analyzer


@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state."""
    with patch.object(st, 'session_state', create=True) as mock_state:
        mock_state.formatted_query = ""
        mock_state.analysis_results = None
        mock_state.selected_query = None
        mock_state.batch_results = []
        mock_state.schema_info = None
        yield mock_state


@pytest.fixture
def sample_sql_content():
    """Sample SQL content for testing."""
    return """
    SELECT *
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_date >= '2023-01-01';

    SELECT DISTINCT c.customer_name, o.order_id
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_items i ON o.order_id = i.order_id
    WHERE i.product_id IN (101, 102, 103);
    """


def test_split_sql_queries(sample_sql_content):
    """Test splitting SQL content into individual queries."""
    queries = split_sql_queries(sample_sql_content)
    assert len(queries) == 2
    assert "SELECT *" in queries[0]
    assert "SELECT DISTINCT" in queries[1]
    assert all(q.strip().endswith(';') for q in queries)


def test_analyze_query_batch(mock_analyzer, sample_sql_content):
    """Test batch analysis of multiple queries."""
    queries = [
        InputAnalysisModel(
            file_name='test1.sql',
            query='SELECT * FROM orders;'
        ),
        InputAnalysisModel(
            file_name='test2.sql',
            query='SELECT id, name FROM customers WHERE age > 25;'
        )
    ]

    results = mock_analyzer.analyze_query(queries)

    assert len(results) == 2
    assert all('filename' in r for r in results)
    assert all('original_query' in r for r in results)
    assert all('analysis' in r for r in results)
    assert mock_analyzer.analyze_query.call_count == 2


def test_batch_analysis_with_schema(mock_analyzer, mock_session_state):
    """Test batch analysis with schema information."""
    schema_info = SchemaInfo(
        table_name="orders",
        columns=[
            {"name": "order_id", "type": "INTEGER"},
            {"name": "customer_id", "type": "INTEGER"},
            {"name": "order_date", "type": "DATE"}
        ],
        row_count=1000
    )

    queries = [
        InputAnalysisModel(
            file_name='test.sql',
            query='SELECT * FROM orders WHERE order_date >= "2023-01-01";'
        )
    ]

    results = mock_analyzer.analyze_query(queries, schema_info=schema_info)

    assert len(results) == 1
    mock_analyzer.analyze_query.assert_called_with(
        queries[0]['query'],
        schema_info=schema_info
    )


def test_batch_analysis_error_handling(mock_analyzer):
    """Test error handling during batch analysis."""

    # Mock analyzer to raise an exception for the second query
    def analyze_side_effect(query, schema_info=None):
        if "error" in query.lower():
            raise Exception("Test error")
        return QueryAnalysis(
            original_query=query,
            category="Performance",
            complexity_score=0.5,
            confidence_score=0.8,
            antipatterns=["Full table scan"],
            suggestions=["Add WHERE clause"],
            optimized_query=query + " WHERE id > 0"
        )

    mock_analyzer.analyze_query.side_effect = analyze_side_effect

    queries = [
        InputAnalysisModel(
            file_name='success.sql',
            query='SELECT * FROM test;'
        ),
        InputAnalysisModel(
            file_name='error.sql',
            query='SELECT error FROM test;'
        )
    ]

    results = mock_analyzer.analyze_query(queries)

    assert len(results) == 1  # Only the successful query should be included
    assert results[0]['filename'] == 'success.sql'


def test_batch_analysis_ui_flow(mock_analyzer, mock_session_state, sample_sql_content):
    """Test the batch analysis UI flow."""
    mock_file = MagicMock()
    mock_file.name = "test.sql"
    mock_file.getvalue.return_value = sample_sql_content.encode()

    with patch('snowflake_optimizer.app.st.file_uploader', return_value=[mock_file]) as mock_uploader, \
            patch('snowflake_optimizer.app.st.button', return_value=True) as mock_button, \
            patch('snowflake_optimizer.app.st.progress', return_value=MagicMock()) as mock_progress, \
            patch('snowflake_optimizer.app.st.empty', return_value=MagicMock()) as mock_empty, \
            patch('snowflake_optimizer.app.st.radio', return_value="Batch Analysis") as mock_radio, \
            patch('snowflake_optimizer.app.st.text_input', return_value="") as mock_text_input, \
            patch('snowflake_optimizer.app.st.text_area', return_value="") as mock_text_area, \
            patch('snowflake_optimizer.app.st.checkbox', return_value=False) as mock_checkbox, \
            patch('snowflake_optimizer.app.st.expander', return_value=MagicMock()) as mock_expander, \
            patch('snowflake_optimizer.app.st.header') as mock_header, \
            patch('snowflake_optimizer.app.st.subheader') as mock_subheader, \
            patch('snowflake_optimizer.app.st.markdown') as mock_markdown, \
            patch('snowflake_optimizer.app.st.code') as mock_code, \
            patch('snowflake_optimizer.app.st.info') as mock_info, \
            patch('snowflake_optimizer.app.st.warning') as mock_warning, \
            patch('snowflake_optimizer.app.st.success') as mock_success, \
            patch('snowflake_optimizer.app.st.error') as mock_error, \
            patch('snowflake_optimizer.app.st.columns', return_value=[MagicMock(), MagicMock()]) as mock_columns:
        # Set up session state
        mock_session_state.formatted_query = ""
        mock_session_state.analysis_results = None
        mock_session_state.selected_query = None
        mock_session_state.batch_results = []

        # Render the view
        render_manual_analysis_view(mock_analyzer)

        # Verify batch analysis was triggered
        mock_header.assert_called_once_with("Manual Query Analysis")
        mock_radio.assert_called_once()

        # Since we're in batch mode, verify file uploader was called
        mock_uploader.assert_called_once_with("Choose SQL files", type=["sql"], accept_multiple_files=True)

        # Verify results were stored in session state
        assert len(mock_session_state.batch_results) > 0

        # Verify the analyzer was called for each query
        expected_calls = len(split_sql_queries(sample_sql_content))
        assert mock_analyzer.analyze_query.call_count == expected_calls


def test_parallel_analysis_performance(mock_analyzer):
    """Test that batch analysis is actually running in parallel."""
    execution_order = []
    execution_times = {}

    def analyze_with_delay(query, schema_info=None):
        query_id = query.split('FROM')[1].strip().rstrip(';')  # Extract test{i} from query
        start_time = time.time()
        execution_order.append(f"{query_id}_start")

        # Simulate varying workloads
        sleep_time = 0.2 if int(query_id.split('test')[1]) % 2 == 0 else 0.1
        time.sleep(sleep_time)

        execution_order.append(f"{query_id}_end")
        execution_times[query_id] = time.time() - start_time

        return QueryAnalysis(
            original_query=query,
            category="Performance",
            complexity_score=0.5,
            confidence_score=0.8,
            antipatterns=["Test pattern"],
            suggestions=["Test suggestion"],
            optimized_query=query
        )

    mock_analyzer.analyze_query.side_effect = analyze_with_delay

    # Create test queries
    queries = [
        {
            'filename': f'test{i}.sql',
            'query': f'SELECT * FROM test{i};'
        }
        for i in range(10)  # 10 queries that would take 2 seconds sequentially
    ]

    # Time the parallel execution
    start_time = time.time()
    results = analyze_query_batch(queries, mock_analyzer)
    total_execution_time = time.time() - start_time

    # Verify results
    assert len(results) == 10
    assert mock_analyzer.analyze_query.call_count == 10

    # Verify parallel execution
    # 1. Total time should be significantly less than sequential time
    sequential_time = sum(execution_times.values())
    assert total_execution_time < sequential_time * 0.5, \
        f"Execution not parallel: total={total_execution_time:.2f}s vs sequential={sequential_time:.2f}s"

    # 2. Check for overlapping executions
    overlapping = False
    starts = {q: i for i, q in enumerate(execution_order) if q.endswith('_start')}
    ends = {q.replace('_end', ''): i for i, q in enumerate(execution_order) if q.endswith('_end')}

    for query1 in starts:
        query1_base = query1.replace('_start', '')
        for query2 in starts:
            if query1 != query2:
                query2_base = query2.replace('_start', '')
                # Check if query2 started before query1 ended
                if starts[query2] > starts[query1] and starts[query2] < ends[query1_base]:
                    overlapping = True
                    break
        if overlapping:
            break

    assert overlapping, "No overlapping query executions detected"

    # 3. Verify concurrent execution through timing analysis
    max_single_time = max(execution_times.values())
    assert total_execution_time < max_single_time * 2, \
        f"Execution might be sequential: total={total_execution_time:.2f}s vs max_single={max_single_time:.2f}s"


def test_parallel_analysis_thread_usage(mock_analyzer):
    """Test that batch analysis uses the correct number of threads."""
    import threading
    active_threads = set()
    thread_lock = threading.Lock()

    def analyze_with_thread_tracking(query, schema_info=None):
        thread_id = threading.get_ident()
        with thread_lock:
            active_threads.add(thread_id)
        time.sleep(0.1)  # Ensure overlap

        result = QueryAnalysis(
            original_query=query,
            category="Performance",
            complexity_score=0.5,
            confidence_score=0.8,
            antipatterns=["Test pattern"],
            suggestions=["Test suggestion"],
            optimized_query=query
        )

        return result

    mock_analyzer.analyze_query.side_effect = analyze_with_thread_tracking

    # Create more queries than default thread pool size
    queries = [
        {
            'filename': f'test{i}.sql',
            'query': f'SELECT * FROM test{i};'
        }
        for i in range(40)  # More than max_workers (32)
    ]

    results = analyze_query_batch(queries, mock_analyzer)

    # Verify results
    assert len(results) == 40
    assert mock_analyzer.analyze_query.call_count == 40

    # Verify thread pool size
    assert len(active_threads) <= 32, f"Used {len(active_threads)} threads, expected <= 32"
    assert len(active_threads) > 1, "Not using multiple threads"


def test_excel_report_creation(mock_analyzer):
    """Test creation of Excel report from batch analysis results."""
    # Create test results
    results = [
        {
            'filename': 'test1.sql',
            'original_query': 'SELECT * FROM table1',
            'analysis': QueryAnalysis(
                original_query='SELECT * FROM table1',
                category='Performance',
                complexity_score=0.7,
                confidence_score=0.8,
                antipatterns=['Full table scan'],
                suggestions=['Add WHERE clause'],
                optimized_query='SELECT * FROM table1 WHERE id > 0'
            )
        },
        {
            'filename': 'test2.sql',
            'original_query': 'SELECT a, b FROM table2',
            'analysis': QueryAnalysis(
                original_query='SELECT a, b FROM table2',
                category='Best Practice',
                complexity_score=0.3,
                confidence_score=0.9,
                antipatterns=['Column wildcards'],
                suggestions=['Specify columns'],
                optimized_query='SELECT id, name FROM table2'
            )
        }
    ]

    # Create Excel report
    excel_data = create_excel_report(results)

    # Verify Excel data was created
    assert isinstance(excel_data, bytes)

    # Read Excel file and verify contents
    import pandas as pd
    import io

    with io.BytesIO(excel_data) as bio:
        # Verify sheets exist
        excel_file = pd.ExcelFile(bio)
        assert 'Errors' in excel_file.sheet_names
        assert 'Metrics' in excel_file.sheet_names
        assert 'Optimizations' in excel_file.sheet_names

        # Read sheets
        errors_df = pd.read_excel(bio, sheet_name='Errors')
        metrics_df = pd.read_excel(bio, sheet_name='Metrics')
        optimizations_df = pd.read_excel(bio, sheet_name='Optimizations')

        # Verify error patterns
        assert len(errors_df) == 2  # Two antipatterns
        assert 'Full table scan' in errors_df['Pattern'].values
        assert 'Column wildcards' in errors_df['Pattern'].values

        # Verify metrics
        assert len(metrics_df) == 2  # Two queries
        assert 0.7 in metrics_df['Complexity'].values
        assert 0.3 in metrics_df['Complexity'].values

        # Verify optimizations
        assert len(optimizations_df) == 2  # Two queries
        assert 'SELECT * FROM table1' in optimizations_df['Original'].values
        assert 'SELECT * FROM table1 WHERE id > 0' in optimizations_df['Optimized'].values


def test_excel_report_error_handling():
    """Test error handling in Excel report creation."""
    # Test with empty results
    with pytest.raises(ValueError, match="No results to export"):
        create_excel_report([])

    # Test with invalid results structure
    invalid_results = [{'invalid': 'structure'}]
    with pytest.raises(Exception):
        create_excel_report(invalid_results)


def test_batch_analysis_export_persistence(mock_analyzer, mock_session_state, sample_sql_content):
    """Test that export functionality doesn't reset analysis results."""
    mock_file = MagicMock()
    mock_file.name = "test.sql"
    mock_file.getvalue.return_value = sample_sql_content.encode()

    with patch('snowflake_optimizer.app.st.file_uploader', return_value=[mock_file]) as mock_uploader, \
            patch('snowflake_optimizer.app.st.button') as mock_button, \
            patch('snowflake_optimizer.app.st.progress', return_value=MagicMock()) as mock_progress, \
            patch('snowflake_optimizer.app.st.empty', return_value=MagicMock()) as mock_empty, \
            patch('snowflake_optimizer.app.st.radio', return_value="Batch Analysis") as mock_radio, \
            patch('snowflake_optimizer.app.st.download_button', return_value=False) as mock_download, \
            patch('snowflake_optimizer.app.st.header') as mock_header, \
            patch('snowflake_optimizer.app.st.subheader') as mock_subheader, \
            patch('snowflake_optimizer.app.st.markdown') as mock_markdown, \
            patch('snowflake_optimizer.app.st.code') as mock_code, \
            patch('snowflake_optimizer.app.st.info') as mock_info, \
            patch('snowflake_optimizer.app.st.warning') as mock_warning, \
            patch('snowflake_optimizer.app.st.success') as mock_success, \
            patch('snowflake_optimizer.app.st.error') as mock_error:
        # Set up mock button responses for first render (analyze)
        mock_button.side_effect = lambda *args, **kwargs: kwargs.get('key') == 'batch_analyze'

        # First render - should trigger analysis
        render_manual_analysis_view(mock_analyzer)

        # Verify analysis was performed
        assert mock_analyzer.analyze_query.call_count == len(split_sql_queries(sample_sql_content))
        assert hasattr(mock_session_state, 'batch_results')
        assert len(mock_session_state.batch_results) > 0

        # Store the analysis results
        first_results = mock_session_state.batch_results.copy()

        # Set up mock responses for second render (export)
        mock_button.side_effect = lambda *args, **kwargs: False  # No buttons clicked
        mock_download.return_value = True  # Download button clicked

        # Second render - should not trigger reanalysis
        render_manual_analysis_view(mock_analyzer)

        # Verify analysis wasn't performed again
        assert mock_analyzer.analyze_query.call_count == len(split_sql_queries(sample_sql_content))

        # Verify results weren't reset
        assert hasattr(mock_session_state, 'batch_results')
        assert len(mock_session_state.batch_results) == len(first_results)

        # Verify export was attempted
        mock_download.assert_called_once()

# ... rest of the file remains unchanged ...
