"""Pytest configuration file."""

import os
import pytest
import streamlit as st
from dotenv import load_dotenv

def check_api_key():
    """Check for API key in Streamlit secrets or environment."""
    try:
        if st.secrets["ANTHROPIC_API_KEY"]:
            return True
    except (KeyError, FileNotFoundError):
        pass
    return bool(os.getenv("ANTHROPIC_API_KEY"))

@pytest.fixture(autouse=True)
def load_env():
    """Load environment variables and check secrets before each test."""
    load_dotenv()
    if not check_api_key():
        pytest.skip("ANTHROPIC_API_KEY not found in secrets.toml or environment") 