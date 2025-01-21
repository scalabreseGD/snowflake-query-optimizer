import io
import json
import traceback
from typing import Optional, List, Dict

import pandas as pd
import streamlit as st

from snowflake_optimizer.connections import initialize_connections
from snowflake_optimizer.query_analyzer import QueryAnalyzer, SchemaInfo
from snowflake_optimizer.utils import format_sql, display_query_comparison, SQL_ANTIPATTERNS, split_sql_queries


def render_manual_analysis_view(page_id: str, analyzer: Optional[QueryAnalyzer]):
    """Render the manual query analysis view."""
    print("\n=== Starting Manual Analysis View ===")
    st.header("Manual Query Analysis")

    if not analyzer:
        st.error("Query analyzer is not initialized. Please check your configuration.")
        return

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

    # Query input methods
    input_method = st.radio(
        "Choose input method",
        ["Direct Input", "File Upload", "Batch Analysis"]
    )

    # Store current batch results
    current_batch_results = st.session_state[f"{page_id}_batch_results"].copy() if hasattr(st.session_state,
                                                                                           'batch_results') else []

    if input_method == "Direct Input":
        st.markdown("### Enter SQL Query")

        # Format button before text area
        if st.button("Format Query"):
            if st.session_state[f"{page_id}_formatted_query"]:
                st.session_state[f"{page_id}_formatted_query"] = format_sql(
                    st.session_state[f"{page_id}_formatted_query"])

        # Text area for SQL input
        query = st.text_area(
            "SQL Query",
            value=st.session_state[f"{page_id}_formatted_query"],
            height=200,
            help="Paste your SQL query here for analysis",
            key="sql_input"
        )

        # Update formatted_query only if query has changed
        if query != st.session_state[f"{page_id}_formatted_query"]:
            st.session_state[f"{page_id}_formatted_query"] = query

        if query:
            st.markdown("### Preview")
            st.code(query, language="sql")

        # Optional schema information
        if st.checkbox("Add table schema information"):
            schema_col1, schema_col2 = st.columns([1, 1])

            with schema_col1:
                table_name = st.text_input("Table name")
                row_count = st.number_input("Approximate row count", min_value=0)

            with schema_col2:
                columns_json = st.text_area(
                    "Columns (JSON format)",
                    help='Example: [{"name": "id", "type": "INTEGER"}, {"name": "email", "type": "VARCHAR"}]'
                )

            try:
                columns = json.loads(columns_json) if columns_json else []
                st.session_state[f"{page_id}_schema_info"] = SchemaInfo(
                    table_name=table_name,
                    columns=columns,
                    row_count=row_count
                )
            except json.JSONDecodeError:
                st.error("Invalid JSON format for columns")
                st.session_state[f"{page_id}_schema_info"] = None

        analyze_button = st.button("Analyze", on_click=lambda: __analyze_query_callback(analyzer))

    elif input_method == "File Upload":
        st.markdown("### Upload SQL File")
        uploaded_file = st.file_uploader("Choose a SQL file", type=["sql"])

        if uploaded_file:
            query = uploaded_file.getvalue().decode()
            st.session_state[f"{page_id}_formatted_query"] = format_sql(query)
            st.markdown("### Preview")
            st.code(st.session_state[f"{page_id}_formatted_query"], language="sql")

            analyze_button = st.button("Analyze", on_click=lambda: __analyze_query_callback(analyzer))

    elif input_method == "Batch Analysis":
        st.markdown("### Upload SQL Files")
        uploaded_files = st.file_uploader("Choose SQL files", type=["sql"], accept_multiple_files=True)

        if uploaded_files:
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process uploaded files
            all_queries = []
            for sql_file in uploaded_files:
                content = sql_file.getvalue().decode()
                queries = split_sql_queries(content)
                print(f"\n=== Splitting SQL Queries ===")
                print(f"Found {len(queries)} queries in {sql_file.name}")
                for i, query in enumerate(queries):
                    print(f"Found valid query of length: {len(query)}")
                    all_queries.append({
                        'filename': f"{sql_file.name} (Query {i + 1})",
                        'query': query
                    })
            print(f"Total queries found: {len(all_queries)}")

            analyze_button = st.button("Analyze All", key="batch_analyze")
            if analyze_button:
                results = []
                for i, query_info in enumerate(all_queries):
                    progress = (i + 1) / len(all_queries)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing {query_info['filename']}...")

                    try:
                        formatted_query = format_sql(query_info['query'])
                        analysis = analyzer.analyze_query(formatted_query)
                        if analysis:
                            results.append({
                                'filename': query_info['filename'],
                                'original_query': query_info['query'],
                                'analysis': analysis
                            })
                    except Exception as e:
                        st.error(f"Failed to analyze {query_info['filename']}: {str(e)}")

                st.session_state[f"{page_id}_batch_results"] = results
                status_text.text("Analysis complete!")

            # Display results if available
            if hasattr(st.session_state, 'batch_results') and st.session_state[f"{page_id}_batch_results"]:
                st.markdown("### Analysis Results")
                for result in st.session_state[f"{page_id}_batch_results"]:
                    with st.expander(f"Results for {result['filename']}"):
                        st.code(result['original_query'], language="sql")
                        st.info(f"Category: {result['analysis'].category}")
                        st.progress(result['analysis'].complexity_score,
                                    text=f"Complexity Score: {result['analysis'].complexity_score:.2f}")

                        if result['analysis'].antipatterns:
                            st.warning("Antipatterns:")
                            for pattern in result['analysis'].antipatterns:
                                st.write(f"- {pattern}")

                        if result['analysis'].suggestions:
                            st.info("Suggestions:")
                            for suggestion in result['analysis'].suggestions:
                                st.write(f"- {suggestion}")

                        if result['analysis'].optimized_query:
                            st.success("Optimized Query:")
                            st.code(result['analysis'].optimized_query, language="sql")

            # Export functionality in a separate section
            if hasattr(st.session_state, 'batch_results') and st.session_state[f"{page_id}_batch_results"]:
                st.markdown("### Export Results")
                try:
                    print("\n=== Export Button Clicked ===")
                    print(f'Creating Excel report for {len(st.session_state[f"{page_id}_batch_results"])} results')
                    excel_data = __create_excel_report(st.session_state[f"{page_id}_batch_results"])
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_data,
                        file_name="query_analysis_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel"  # Add unique key
                    )
                except Exception as e:
                    st.error(f"Failed to create Excel report: {str(e)}")
                    print(f"Excel export error: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")

    # Display analysis results if available
    if st.session_state[f"{page_id}_analysis_results"]:
        st.subheader("Analysis Results")

        # Display query category and complexity
        st.info(f'Query Category: {st.session_state[f"{page_id}_analysis_results"].category}')
        st.progress(st.session_state[f"{page_id}_analysis_results"].complexity_score,
                    text=f'Complexity Score: {st.session_state[f"{page_id}_analysis_results"].complexity_score:.2f}')

        # Display antipatterns
        if st.session_state[f"{page_id}_analysis_results"].antipatterns:
            st.warning("Antipatterns Detected:")
            for pattern in st.session_state[f"{page_id}_analysis_results"].antipatterns:
                st.write(f"- {pattern}")

        # Display suggestions
        if st.session_state[f"{page_id}_analysis_results"].suggestions:
            st.info("Optimization Suggestions:")
            for suggestion in st.session_state[f"{page_id}_analysis_results"].suggestions:
                st.write(f"- {suggestion}")

        # Display optimized query
        if st.session_state[f"{page_id}_analysis_results"].optimized_query:
            st.success("Query Optimization Results")
            display_query_comparison(
                st.session_state[f"{page_id}_formatted_query"],
                st.session_state[f"{page_id}_analysis_results"].optimized_query
            )
            if st.button("Copy Optimized Query"):
                st.session_state[f"{page_id}_clipboard"] = format_sql(
                    st.session_state[f"{page_id}_analysis_results"].optimized_query)
                st.success("Query copied to clipboard!")


def __analyze_query_callback(analyzer: Optional[QueryAnalyzer]):
    """Callback function for analyzing queries in the manual analysis view.

    Args:
        analyzer: QueryAnalyzer instance to use for analysis
    """
    print("\n=== Starting Query Analysis Callback ===")

    if not analyzer:
        print("No analyzer available")
        st.error("Query analyzer is not initialized")
        return

    if not st.session_state[f"{page_id}_formatted_query"]:
        print("No query to analyze")
        return

    try:
        print(f'Analyzing query of length: {len(st.session_state[f"{page_id}_formatted_query"])}')
        st.session_state[f"{page_id}_analysis_results"] = analyzer.analyze_query(
            st.session_state[f"{page_id}_formatted_query"],
            schema_info=st.session_state[f"{page_id}_schema_info"] if hasattr(st.session_state, 'schema_info') else None
        )
        print("Analysis completed successfully")

        # Store the analyzed query for comparison
        st.session_state[f"{page_id}_selected_query"] = st.session_state[f"{page_id}_formatted_query"]

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        st.error(f"Analysis failed: {str(e)}")
        st.session_state[f"{page_id}_analysis_results"] = None


def __create_excel_report(batch_results: List[Dict]) -> bytes:
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

                for category, patterns in SQL_ANTIPATTERNS.items():
                    for code, details in patterns.items():
                        if any(detect.lower() in pattern.lower() for detect in details['detection']):
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
                        'Details': pattern,
                        'Suggestion': analysis.suggestions[0] if analysis.suggestions else 'None'
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
                        'Suggestion': analysis.suggestions[0] if analysis.suggestions else 'None'
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
                columns=['Query', 'Pattern Code', 'Pattern Name', 'Category', 'Description', 'Impact', 'Details',
                         'Suggestion']).to_excel(writer, sheet_name='Errors', index=False)

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


def main():
    st.title("Snowflake Query Optimizer")
    st.write("Analyze and optimize your Snowflake SQL queries")
    page_id = 'manual_analysis'
    # Initialize connections
    _collector, _analyzer = initialize_connections(page_id)
    render_manual_analysis_view(page_id, _analyzer)


main()
