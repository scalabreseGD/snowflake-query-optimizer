import difflib
import logging
from typing import List

import sqlparse
import streamlit as st

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


def init_common_states(page_id):
    # Initialize session state variables if they don't exist
    if f"{page_id}_formatted_query" not in st.session_state:
        st.session_state[f"{page_id}_formatted_query"] = ""
    if f"{page_id}_analysis_results" not in st.session_state:
        st.session_state[f"{page_id}_analysis_results"] = None
    if f"{page_id}_selected_query" not in st.session_state:
        st.session_state[f"{page_id}_selected_query"] = None
    if f"{page_id}_batch_results" not in st.session_state:
        st.session_state[f"{page_id}_batch_results"] = []
    if f"{page_id}_schema_info" not in st.session_state:
        st.session_state[f"{page_id}_schema_info"] = None
    if f"{page_id}_clipboard" not in st.session_state:
        st.session_state[f"{page_id}_clipboard"] = None


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
