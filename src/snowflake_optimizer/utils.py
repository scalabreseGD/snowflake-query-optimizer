import difflib
import io
import logging
import traceback
from typing import List, Dict

import pandas as pd
import sqlparse
import streamlit as st

from snowflake_optimizer.data_collector import SnowflakeQueryExecutor
from snowflake_optimizer.models import OutputAnalysisModel

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
        },
        'NIN001': {
            'name': 'Not IN Subquery',
            'description': 'Using NOT IN with a subquery can lead to performance issues',
            'impact': 'Medium',
            'detection': ['NOT IN (SELECT ...)', 'subquery in NOT IN']
        },
        'ORC001': {
            'name': 'Overuse of OR Conditions',
            'description': 'Excessive OR conditions can prevent index usage',
            'impact': 'Medium',
            'detection': ['multiple OR in WHERE clause']
        },
        'IMP001': {
            'name': 'Implicit Data Type Conversion',
            'description': 'Implicit conversions can lead to index scans instead of seeks',
            'impact': 'High',
            'detection': ['data type mismatch in WHERE clause']
        },
        'NJN001': {
            'name': 'Nested Joins',
            'description': 'Deeply nested joins can complicate execution plans',
            'impact': 'Medium',
            'detection': ['multiple nested JOINs']
        },
        'SRT001': {
            'name': 'Unnecessary Sorting',
            'description': 'Sorting data unnecessarily increases query time',
            'impact': 'Low',
            'detection': ['ORDER BY without need']
        },
        'AGG001': {
            'name': 'Unnecessary Aggregation',
            'description': 'Aggregating data without necessity adds overhead',
            'impact': 'Low',
            'detection': ['GROUP BY without need']
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
        },
        'DUP001': {
            'name': 'Duplicate Rows',
            'description': 'Lack of constraints leading to duplicate data entries',
            'impact': 'High',
            'detection': ['no UNIQUE constraint', 'no PRIMARY KEY']
        },
        'UDF001': {
            'name': 'Improper Use of User-Defined Functions',
            'description': 'Using UDFs in WHERE clauses can hinder performance',
            'impact': 'Medium',
            'detection': ['UDF in WHERE clause']
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
        },
        'CTE001': {
            'name': 'Overuse of CTEs',
            'description': 'Excessive Common Table Expressions can reduce readability',
            'impact': 'Low',
            'detection': ['multiple CTEs in query']
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
        },
        'HNT001': {
            'name': 'Use of Hints',
            'description': 'Over-reliance on optimizer hints can reduce portability',
            'impact': 'Low',
            'detection': ['USE INDEX', 'FORCE INDEX']
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
        },
        'EXE001': {
            'name': 'Execution of Untrusted Scripts',
            'description': 'Running scripts from unverified sources',
            'impact': 'High',
            'detection': ['EXECUTE from external source']
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
        },
        'MAG001': {
            'name': 'Magic Numbers',
            'description': 'Use of unexplained numeric literals in queries',
            'impact': 'Low',
            'detection': ['hard-coded numbers']
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


def create_results_expanders(executor: SnowflakeQueryExecutor, results: List[OutputAnalysisModel]):
    for result in results:
        with st.expander(f"Results for {result['filename']}", expanded=len(results) == 1):
            st.code(result['original_query'], language="sql")
            logging.debug(f'Analysis results - Category: {result["analysis"].category}, '
                          f'Complexity: {result["analysis"].complexity_score:.2f}')
            st.info(f"Category: {result['analysis'].category}")
            st.progress(result['analysis'].complexity_score,
                        text=f"Complexity Score: {result['analysis'].complexity_score:.2f}")

            if result['analysis'].antipatterns:
                logging.debug(f'Antipatterns detected: {len(result["analysis"].antipatterns)}')
                st.warning("Antipatterns Detected:")
                for pattern in result['analysis'].antipatterns:
                    st.write(f"- {pattern}")

            if result['analysis'].suggestions:
                logging.debug(f'Optimization suggestions: {len(result["analysis"].suggestions)}')
                st.info('Optimization Suggestions:')
                for suggestion in result['analysis'].suggestions:
                    st.write(f"- {suggestion}")

            if result['analysis'].optimized_query:
                logging.info("Optimized query generated")
                st.success("Optimized Query:")

                display_query_comparison(
                    executor,
                    result.original_query,
                    result.analysis.optimized_query
                )


def create_export_excel_from_results(results: List[OutputAnalysisModel]):
    st.markdown("### Export Results")
    try:
        print("\n=== Export Button Clicked ===")
        print(f'Creating Excel report for {len(results)} results')
        st.download_button(
            label="Download Excel Report",
            data=__create_excel_report(results),
            file_name="query_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"  # Add unique key
        )
    except Exception as e:
        st.error(f"Failed to create Excel report: {str(e)}")
        print(f"Excel export error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")


def __create_excel_report(batch_results: List[OutputAnalysisModel]) -> bytes:
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
                category = None
                for category, patterns in SQL_ANTIPATTERNS.items():
                    for code, details in patterns.items():
                        if code.lower() in pattern.lower():
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
                        'Details': pattern
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
                columns=['Query', 'Pattern Code', 'Pattern Name', 'Category', 'Description', 'Impact',
                         'Details', ]).to_excel(writer, sheet_name='Errors', index=False)

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


def display_query_comparison(executor: SnowflakeQueryExecutor, original: str, optimized: str):
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
    result_columns = st.columns([0.1, 0.8, 0.1])
    with result_columns[1]:
        if st.button('Compare Original and Optimized'):
            original_query_df, optimized_query_df, difference_df = executor.compare_optimized_query_with_original(
                optimized_query=optimized,
                original_query=original)
            show_performance_difference(original_query_df, optimized_query_df, difference_df)

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


def show_performance_difference(original_query_df: pd.DataFrame, optimized_query_df: pd.DataFrame,
                                difference_df: pd.DataFrame):
    st.markdown("### Performance Difference")
    st.markdown("### Original Query")
    st.dataframe(original_query_df)
    st.markdown("### Optimized Query")
    st.dataframe(optimized_query_df)

    difference_records = difference_df.to_dict(orient='records')[0]
    for column_name, column_value in difference_records.items():
        if column_value < 0:
            st.success(f"{column_name}: {column_value}")
        elif column_value == 0:
            st.warning(f"{column_name}: {column_value}")
        else:
            st.error(f"{column_name}: {column_value}")


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
