"""Tests for the query analyzer module."""

import os
import pytest
import streamlit as st
from anthropic import AuthenticationError
from snowflake_optimizer.query_analyzer import QueryAnalyzer

def get_api_key():
    """Get API key from either Streamlit secrets or environment variables."""
    # Try Streamlit secrets first
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variable
        return os.getenv("ANTHROPIC_API_KEY")

def test_api_key_authentication():
    """Test that the API key authentication works correctly."""
    # Get API key
    api_key = get_api_key()
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in secrets.toml or environment")
    
    # Test with invalid key first
    with pytest.raises(AuthenticationError):
        analyzer = QueryAnalyzer(anthropic_api_key="invalid_key")
        analyzer.analyze_query("SELECT * FROM test")

    # Test with actual key
    try:
        analyzer = QueryAnalyzer(anthropic_api_key=api_key)
        # Simple query to test
        result = analyzer.analyze_query("SELECT * FROM test")
        assert result is not None
    except AuthenticationError as e:
        pytest.fail(f"Authentication failed with configured API key: {str(e)}")

def test_api_key_format():
    """Test that the API key follows the expected format."""
    api_key = get_api_key()
    
    # Check if API key is not empty
    assert api_key, "ANTHROPIC_API_KEY not found in secrets.toml or environment"
    
    # Check if API key starts with expected prefix
    assert api_key.startswith("sk-ant-"), "API key should start with 'sk-ant-'"
    
    # Check minimum length (typical Anthropic API keys are quite long)
    assert len(api_key) > 20, "API key seems too short" 