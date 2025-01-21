"""Main Streamlit application for Snowflake Query Optimizer."""

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from snowflake_optimizer.connections import setup_logging, initialize_connections
from snowflake_optimizer.query_analyzer import QueryAnalyzer, SchemaInfo
from snowflake_optimizer.utils import format_sql, display_query_comparison

# Set up logging
setup_logging()

# Load environment variables
load_dotenv()

if "manual_queries" not in st.session_state:
    st.session_state.manual_queries = []
if "schema_info" not in st.session_state:
    st.session_state.schema_info = None


def render_advanced_optimization_view(analyzer: QueryAnalyzer):
    """Render the advanced optimization view."""
    st.markdown("## Advanced Optimization Mode")

    # Input section
    st.markdown("### Query Input")
    query = st.text_area("Enter your SQL query", height=200, key="advanced_sql_input")

    col1, col2 = st.columns(2)
    with col1:
        schema_info = st.text_area("Table Schema Information (Optional)",
                                   placeholder="Enter table definitions, indexes, etc.",
                                   height=150)
    with col2:
        partition_info = st.text_area("Partitioning Details (Optional)",
                                      placeholder="Enter partitioning strategy details",
                                      height=150)

    # Analysis options
    st.markdown("### Optimization Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        analyze_clustering = st.checkbox("Analyze Clustering Keys", value=True)
        suggest_materialization = st.checkbox("Suggest Materialization", value=True)
    with col2:
        analyze_search = st.checkbox("Analyze Search Optimization", value=True)
        suggest_caching = st.checkbox("Suggest Caching Strategy", value=True)
    with col3:
        analyze_partitioning = st.checkbox("Analyze Partitioning", value=True)

    if st.button("Analyze Query"):
        if query:
            try:
                with st.spinner("Analyzing query..."):
                    result = analyzer.analyze_query(
                        query,
                        schema_info=schema_info if schema_info else None,
                        partition_info=partition_info if partition_info else None,
                        analyze_clustering=analyze_clustering,
                        suggest_materialization=suggest_materialization,
                        analyze_search=analyze_search,
                        suggest_caching=suggest_caching,
                        analyze_partitioning=analyze_partitioning
                    )

                    if result:
                        st.markdown("### Analysis Results")

                        # Query Information
                        st.markdown("#### Query Information")
                        st.markdown(f"**Category:** {result.category}")
                        st.markdown(f"**Confidence Score:** {result.confidence_score:.2f}")

                        # Display comparisons
                        display_query_comparison(query, result.optimized_query)

                        # Antipatterns
                        if result.antipatterns:
                            st.markdown("#### Detected Antipatterns")
                            for pattern in result.antipatterns:
                                st.warning(pattern)

                        # Optimization suggestions
                        if result.suggestions:
                            st.markdown("#### Optimization Suggestions")
                            for suggestion in result.suggestions:
                                st.info(suggestion)
                    else:
                        st.error("Failed to analyze query. Please try again.")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
        else:
            st.warning("Please enter a SQL query to analyze.")


def analyze_query_with_retry(analyzer: QueryAnalyzer, query: str, schema_info: Optional[SchemaInfo] = None,
                             max_retries: int = 3) -> Optional[Any]:
    """Analyze a query with retry logic and error handling.
    
    Args:
        analyzer: QueryAnalyzer instance
        query: SQL query to analyze
        schema_info: Optional schema information
        max_retries: Maximum number of retry attempts
        
    Returns:
        Analysis results or None if analysis fails
    """
    print(f"\n=== Analyzing Query (length: {len(query)}) ===")

    for attempt in range(max_retries):
        try:
            # Try to format the query first
            formatted_query = format_sql(query)
            print(f"Attempt {attempt + 1}: Formatted query length: {len(formatted_query)}")

            # Analyze the formatted query
            result = analyzer.analyze_query(
                formatted_query,
                schema_info=schema_info
            )
            print("Analysis successful")
            return result

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to analyze query after {max_retries} attempts: {str(e)}")
            continue

    return None


def group_related_queries(queries: List[Dict]) -> List[Dict]:
    """Group related queries based on common tables and patterns.
    
    Args:
        queries: List of query dictionaries with analysis results
        
    Returns:
        List of query groups with optimization suggestions
    """
    groups = []
    processed = set()

    for i, query in enumerate(queries):
        if i in processed:
            continue

        related = {i}
        base_tables = set(query.get('tables', []))

        # Find related queries
        for j, other in enumerate(queries):
            if j != i and j not in processed:
                other_tables = set(other.get('tables', []))
                # If queries share tables or have similar patterns
                if (base_tables & other_tables) or (
                        query['analysis'].category == other['analysis'].category
                ):
                    related.add(j)

        # Create group
        group = {
            'queries': [queries[idx] for idx in related],
            'common_tables': base_tables,
            'category': query['analysis'].category,
            'group_suggestions': []
        }

        # Aggregate group-level suggestions
        all_suggestions = []
        for q in group['queries']:
            if q['analysis'].suggestions:
                all_suggestions.extend(q['analysis'].suggestions)

        # Find common suggestions
        if all_suggestions:
            from collections import Counter
            suggestion_counts = Counter(all_suggestions)
            group['group_suggestions'] = [
                sugg for sugg, count in suggestion_counts.items()
                if count > 1  # Suggestion appears in multiple queries
            ]

        groups.append(group)
        processed.update(related)

    return groups


def get_error_analysis_prompt(query: str) -> str:
    """Generate the prompt for LLM to analyze SQL query errors and anti-patterns.
    
    Args:
        query: SQL query to analyze
        
    Returns:
        Formatted prompt for LLM
    """
    return f"""Analyze the following SQL query for potential errors, anti-patterns, and optimization opportunities. 
Provide a detailed analysis in the following JSON format:

{{
    "error_patterns": [
        {{
            "category": "One of: PERFORMANCE, DATA_QUALITY, COMPLEXITY, BEST_PRACTICE, SECURITY, MAINTAINABILITY",
            "code": "One of: FTS001, IJN001, IDX001, LDT001, NUL001, DTM001, NSQ001, CJN001, WCD001, ALS001, INJ001, PRM001, CMT001, FMT001",
            "pattern": "Name of the anti-pattern",
            "severity": "One of: High, Medium, Low",
            "impact": "Brief impact description",
            "location": "Where in the query this occurs",
            "description": "Detailed explanation of the issue",
            "recommendation": "How to fix or improve"
        }}
    ],
    "query_metrics": {{
        "complexity_score": "Score from 0-1",
        "readability_score": "Score from 0-1",
        "maintainability_score": "Score from 0-1",
        "performance_impact": "Score from 0-1",
        "structural_metrics": {{
            "num_joins": "Number",
            "num_conditions": "Number",
            "nesting_depth": "Number",
            "num_aggregations": "Number"
        }}
    }}
}}

Anti-pattern Categories and Codes:
1. PERFORMANCE:
   - FTS001: Full Table Scan - Query performs a full table scan without appropriate filtering
   - IJN001: Inefficient Join - Join conditions missing or using non-indexed columns
   - IDX001: Missing Index - Frequently filtered or joined columns lack appropriate indexes
   - LDT001: Large Data Transfer - Query retrieves excessive data volume

2. DATA_QUALITY:
   - NUL001: Unsafe Null Handling - Improper handling of NULL values in comparisons
   - DTM001: Data Type Mismatch - Implicit data type conversions in comparisons

3. COMPLEXITY:
   - NSQ001: Nested Subquery - Deeply nested subqueries that could be simplified
   - CJN001: Complex Join Chain - Long chain of joins that could be simplified

4. BEST_PRACTICE:
   - WCD001: Wildcard Column Usage - Using SELECT * instead of specific columns
   - ALS001: Missing Table Alias - Tables or subqueries without clear aliases

5. SECURITY:
   - INJ001: SQL Injection Risk - Potential SQL injection vulnerabilities
   - PRM001: Missing Parameterization - Hard-coded values instead of parameters

6. MAINTAINABILITY:
   - CMT001: Missing Comments - Complex logic without explanatory comments
   - FMT001: Poor Formatting - Inconsistent or poor SQL formatting

Focus on:
1. Performance implications
2. Logical correctness
3. Best practices
4. Maintainability issues
5. Security concerns
6. Complexity assessment

Query to analyze:
{query}

Provide the analysis in the exact JSON format specified above."""


def analyze_query_batch(queries: List[Dict],
                        analyzer: QueryAnalyzer,
                        schema_info: Optional[SchemaInfo] = None) -> List[Dict]:
    """Analyze a batch of queries in parallel using multi-threading.
    
    Args:
        queries: List of query dictionaries containing filename and query
        analyzer: QueryAnalyzer instance
        schema_info: Optional schema information
        
    Returns:
        List of analysis results
    """
    print("\n=== Starting Batch Analysis ===")
    print(f"Number of queries to analyze: {len(queries)}")
    results = []

    # Calculate optimal number of workers
    max_workers = min(32, len(queries))  # Cap at 32 threads
    print(f"Using {max_workers} worker threads")

    def analyze_single_query(query_info: Dict) -> Optional[Dict]:
        try:
            print(f"\nAnalyzing query from {query_info['filename']}")
            analysis_result = analyzer.analyze_query(
                query_info['query'],
                schema_info=schema_info
            )

            if analysis_result:
                print(f"Analysis successful for {query_info['filename']}")
                return {
                    "filename": query_info['filename'],
                    "original_query": query_info['query'],
                    "analysis": analysis_result
                }
            print(f"No analysis result for {query_info['filename']}")
            return None

        except Exception as e:
            print(f"Error analyzing {query_info['filename']}: {str(e)}")
            return None

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and get futures
        future_to_query = {
            executor.submit(analyze_single_query, query_info): query_info
            for query_info in queries
        }

        # Process completed futures as they finish
        for future in as_completed(future_to_query):
            query_info = future_to_query[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"Added result for {query_info['filename']} to results list")
            except Exception as e:
                print(f"Analysis failed for {query_info['filename']}: {str(e)}")

    print(f"\nBatch analysis completed. Total results: {len(results)}")
    return results


def compare_query(collector):
    query_id = '01b9d4ef-0a08-cfb7-0000-5f21c984fa8e'
    optimized = """
    SELECT count(*) AS cnt,
       'CRM' AS env
FROM
    (SELECT DISTINCT 'arbys' AS brand_id,
                     'sfmc' AS source_system_name,
                     account_id,
                     job_id,
                     subscriber_key,
                     external_id,
                     customer_id,
                     offer_id,
                     batch_id,
                     list_id,
                     creative_variant,
                     event_date,
                     offer_code,
                     sub_id,
                     triggered_send_id,
                     error_code,
                     datasource_name,
                     email_address,
                     offer_name,
                     parent_offer_id,
                     journey_name,
                     journey_step,
                     user_defined_segment_1,
                     user_defined_segment_2,
                     user_defined_segment_3,
                     user_defined_segment_4,
                     user_defined_segment_5,
                     offer_decision_logic,
                     email_name,
                     campaign_id,
                     to_varchar(mdt_created_on, 'YYYYMMDD')::INTEGER AS load_id,
                     split_part(mdt_filename, '/', -1) AS load_filename
     FROM crm.arbys.sfmc_sendlog)
UNION ALL
SELECT count(*) AS cnt,
       'IDS' AS env
FROM
    (SELECT DISTINCT brand_id,
                     source_system_name,
                     account_id,
                     job_id,
                     subscriber_key,
                     profile_id,
                     external_id,
                     customer_id,
                     offer_id,
                     batch_id,
                     list_id,
                     creative_variant,
                     event_dttm,
                     offer_code,
                     subscriber_id,
                     triggered_send_external_key,
                     error_code,
                     datasource_name,
                     email_address,
                     offer_name,
                     parent_offer_id,
                     journey_name,
                     journey_step,
                     strength_of_customer,
                     user_defined_segment_1,
                     user_defined_segment_2,
                     user_defined_segment_3,
                     user_defined_segment_4,
                     user_defined_segment_5,
                     offer_decision_logic,
                     offer_1_decision_logic,
                     point_balance,
                     email_name,
                     campaign_id,
                     campaign_name,
                     campaign_type,
                     campaign_category_type,
                     campaign_description,
                     campaign_objective,
                     campaign_start_date,
                     campaign_end_date,
                     campaign_duration,
                     load_filename
     FROM ids_qa.cust.crm_campaign_send_log_arbys);
    """
    collector.compare_optimized_query_with_original(optimized, query_id)


def main():
    """Main function to run the Streamlit application."""
    logging.info("Starting main application")
    st.title("Snowflake Query Optimizer")
    st.write("Analyze and optimize your Snowflake SQL queries")

    # Initialize connections
    collector, analyzer = initialize_connections('app')
    # compare_query(collector)

    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["Query History Analysis", "Manual Analysis", "Advanced Optimization"]
    )
    logging.info(f"Selected mode: {mode}")

    render_advanced_optimization_view(analyzer)

    logging.debug("Main application loop completed")


if __name__ == "__main__":
    main()
