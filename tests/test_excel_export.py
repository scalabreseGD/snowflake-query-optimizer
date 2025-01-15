import pytest
import pandas as pd
from io import BytesIO
from snowflake_optimizer.app import create_excel_report
from snowflake_optimizer.query_analyzer import QueryAnalysis

def test_create_excel_report():
    # Create mock analysis results
    mock_results = [
        {
            "filename": "test1.sql",
            "original_query": "SELECT * FROM table1",
            "analysis": QueryAnalysis(
                original_query="SELECT * FROM table1",
                category="Performance",
                complexity_score=0.5,
                confidence_score=0.8,
                antipatterns=["Full table scan"],
                suggestions=["Add WHERE clause"],
                optimized_query="SELECT * FROM table1 WHERE id > 0"
            )
        },
        {
            "filename": "test2.sql",
            "original_query": "SELECT a, b FROM table2",
            "analysis": QueryAnalysis(
                original_query="SELECT a, b FROM table2",
                category="Optimization",
                complexity_score=0.3,
                confidence_score=0.9,
                antipatterns=[],
                suggestions=["No improvements needed"],
                optimized_query=None
            )
        }
    ]
    
    # Call the export function
    excel_data = create_excel_report(mock_results)
    
    # Verify the output is bytes
    assert isinstance(excel_data, bytes)
    assert len(excel_data) > 0
    
    # Read the Excel file and verify its contents
    with BytesIO(excel_data) as bio:
        # Read all sheets into a dict of dataframes
        dfs = pd.read_excel(bio, sheet_name=None)
        
        # Verify all expected sheets exist
        assert set(dfs.keys()) == {'Errors', 'Metrics', 'Optimizations'}
        
        # Verify Errors sheet
        errors_df = dfs['Errors']
        assert 'Query' in errors_df.columns
        assert 'Pattern' in errors_df.columns
        assert len(errors_df) == 1  # Only test1.sql has an antipattern
        
        # Verify Metrics sheet
        metrics_df = dfs['Metrics']
        assert 'Query' in metrics_df.columns
        assert 'Complexity' in metrics_df.columns
        assert len(metrics_df) == 2  # Both queries should have metrics
        
        # Verify Optimizations sheet
        opt_df = dfs['Optimizations']
        assert 'Query' in opt_df.columns
        assert 'Original' in opt_df.columns
        assert 'Optimized' in opt_df.columns
        assert len(opt_df) == 2  # Both queries should be in optimizations 