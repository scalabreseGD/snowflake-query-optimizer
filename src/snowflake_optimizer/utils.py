import difflib
import hashlib
import io
import logging
import traceback
import uuid
import re
from typing import List, Dict, Optional
import datetime
from decimal import Decimal

import pandas as pd
import sqlparse
import streamlit as st

from snowflake_optimizer.constants import SQL_ANTIPATTERNS
from snowflake_optimizer.data_collector import SnowflakeQueryExecutor
from snowflake_optimizer.models import OutputAnalysisModel, SchemaInfo
from snowflake_optimizer.query_analyzer import QueryAnalyzer


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


def __build_affected_objects(schema_info: Optional[List[SchemaInfo]] = None):
    if schema_info:
        affected_objects_md = ("Affected Objects:\n"
                               "| TABLE NAME | ROW COUNT | SIZE IN BYTES |\n"
                               "|------------|-----------|---------------|\n")
        for info in schema_info:
            affected_objects_md += f"|{info.table_name}|{info.row_count}|{info.size_bytes}|\n"
        st.markdown(affected_objects_md)


def create_results_expanders(executor: SnowflakeQueryExecutor, results: List[OutputAnalysisModel], original_query_history: Optional[pd.Series] = pd.Series([])):
    for result in results:
        with st.expander(f"Results for {result['filename']}", expanded=len(results) == 1):
            st.code(result['original_query'], language="sql")
            __build_affected_objects(schema_info=result.schema_info)
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
                    result.analysis.optimized_query,
                    original_query_history
                )


def create_export_excel_from_results(results: List[OutputAnalysisModel]):
    st.markdown("### Export Results")
    try:
        print("\n=== Export Button Clicked ===")
        print(f'Creating Excel report for {len(results)} results')
        st.download_button(
            label="Download Excel Report",
            data=create_excel_report(results),
            file_name="query_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"  # Add unique key
        )
    except Exception as e:
        st.error(f"Failed to create Excel report: {str(e)}")
        print(f"Excel export error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")


def create_excel_report(batch_results: List[OutputAnalysisModel]) -> bytes:
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


def display_query_comparison(executor: SnowflakeQueryExecutor, original: str, optimized: str, original_query_history: Optional[pd.Series] = pd.Series([])):
    """Display a side-by-side comparison of original and optimized queries."""
    if not original or not optimized:
        print("Missing query for comparison!")
        return

    st.markdown("### Query Comparison")

    # Create columns for side-by-side view
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Query**")
        if original:
            formatted_original = format_sql(original)
        else:
            formatted_original = format_sql(original_query_history['query_text'])
        st.code(formatted_original, language="sql")

    with col2:
        st.markdown("**Optimized Query**")
        formatted_optimized = format_sql(optimized)
        st.code(formatted_optimized, language="sql")
    result_columns = st.columns([0.1, 0.8, 0.1])
    with result_columns[1]:
        waiting_time_in_seconds = st.slider(label="Comparing timeout in seconds. 0 is no timeout", min_value=0,
                                            max_value=3600, value=600, key=hashlib.sha256(original.encode()).hexdigest()[:16])
        if st.button('Compare Original and Optimized', key=hashlib.sha256(original.encode()).hexdigest()[:32]):
            if is_safe_select_query(optimized):
                with st.spinner("Comparing original and optimized queries..."):
                    original_query_df, optimized_query_df, difference_df = executor.compare_optimized_query_with_original(
                        optimized_query=optimized,
                        original_query=original,
                        original_query_history=original_query_history,
                        waiting_timeout_in_secs=waiting_time_in_seconds if waiting_time_in_seconds != 0 else None
                    )
                show_performance_difference(original_query_df, optimized_query_df, difference_df)
            else:
                st.error("Possible SQL injection. Check optimized query. Only SELECT statements are allowed")

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
    
    minimum_expected_columns = ['EXECUTION_TIME_SECONDS', 'MB_SCANNED', 'ROWS_PRODUCED', 'COMPILATION_TIME_SECONDS',
                                'CREDITS_USED_CLOUD_SERVICES']
    st.markdown("### Original Query")
    st.dataframe(format_time_columns(original_query_df))
    st.markdown("### Optimized Query")
    st.dataframe(format_time_columns(optimized_query_df))

    st.markdown("### Performance Difference")
    difference_records = difference_df.to_dict(orient='records')[0]
    if all([key in minimum_expected_columns for key in difference_records.keys()]):
        for column_name, column_value in difference_records.items():
            if column_name in ["EXECUTION_TIME_SECONDS", "COMPILATION_TIME_SECONDS"]:
                print(f"Original query df: {optimized_query_df}")
                column_value_formatted = f"{column_value/original_query_df[column_name].iloc[0]: .2%}"
                column_name = column_name.replace("_SECONDS",'_%')
            elif column_name == "CREDITS_USED_CLOUD_SERVICES":
                column_value_formatted = f"{column_value/original_query_df[column_name].iloc[0]: .2%}"
                column_name = "CREDITS_USED_CLOUD_SERVICES" + "_%"
            elif column_name in ["MB_SCANNED", "ROWS_PRODUCED"]:
                try:
                    column_value_formatted=int(column_value)
                except:
                    column_value_formatted=column_value
            else:
                column_value_formatted = column_value

            if column_value > 0:
                if column_name != 'ROWS_PRODUCED':
                    st.success(f"{column_name}: {column_value_formatted}")
                else:
                    st.error(f"{column_name}: {column_value_formatted}")
            elif column_value == 0:
                st.warning(f"{column_name}: {column_value_formatted}")
            else:
                st.error(f"{column_name}: {column_value_formatted}")
    else:
        raise ValueError(f'{minimum_expected_columns} is missing in {difference_records.keys()}')


def evaluate_or_repair_query(output_analysis: OutputAnalysisModel,
                             analyzer: QueryAnalyzer,
                             executor: SnowflakeQueryExecutor):
    query = output_analysis.analysis.optimized_query
    error_message = executor.compile_query(query)
    if error_message:
        repaired_query = analyzer.repair_query(query=query, error_message=error_message)
        output_analysis.analysis.optimized_query = repaired_query
    return output_analysis


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

def format_time_columns(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy() 
    columns_mapping = {
        'EXECUTION_TIME_SECONDS': 'EXECUTION_TIME',
        'COMPILATION_TIME_SECONDS': 'COMPILATION_TIME'
    }
    new_columns = []
    for old_col, new_col in columns_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col].map(lambda a: str(datetime.timedelta(seconds=a))[:10])
            df.drop(columns=[old_col], inplace=True)
            new_columns.append(new_col)
    
    #Reorder columns to have time at the beginning for better visibility
    df = df[[col for col in new_columns if col in df.columns] + [col for col in df.columns if col not in new_columns]]

    return df

def is_safe_select_query(query: str) -> bool:
    """
    Checks if the given Snowflake SQL query consists only of SELECT statements
    and prevents SQL injection by disallowing other SQL commands.
    
    :param query: The SQL query string to check.
    :return: True if the query is safe and contains only SELECT statements, False otherwise.
    """
    # Normalize query (strip spaces and convert to lowercase for checking)
    query = query.strip().lower()
    
    # Disallow semicolons to prevent stacked queries
    if ";" in query:
        return False
    
    # Match only SELECT or WITH statements using regex
    select_pattern = re.compile(r"^(\s*select\s+|\s*with\s+)", re.IGNORECASE)
    
    # Ensure it starts with SELECT or WITH and does not contain forbidden SQL keywords
    forbidden_keywords = [
        "insert", "update", "delete", "drop", "alter", "truncate", "create", "exec", "execute",
        "merge", "grant", "revoke", "call", "begin", "commit", "rollback"
    ]
    
    # Check if it starts with SELECT or WITH
    if not select_pattern.match(query):
        return False
    
    # Check for forbidden keywords
    for keyword in forbidden_keywords:
        if re.search(rf"\b{keyword}\b", query, re.IGNORECASE):
            return False
    
    return True
