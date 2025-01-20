"""Module for analyzing and optimizing SQL queries using LLMs."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import sqlparse
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
from sqlglot import parse_one, exp


class QueryCategory(str, Enum):
    """Enumeration of query categories."""

    DATA_MANIPULATION = "Data Manipulation"
    REPORTING = "Reporting"
    ETL = "ETL"
    ANALYTICS = "Analytics"
    UNKNOWN = "Unknown"


@dataclass
class SchemaInfo:
    """Contains table schema and statistics information."""

    table_name: str
    columns: List[Dict[str, str]]
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    indexes: Optional[List[str]] = None
    partitioning: Optional[Dict[str, str]] = None


@dataclass
class QueryAnalysis:
    """Contains the analysis results for a query."""

    original_query: str
    optimized_query: Optional[str]
    antipatterns: List[str]
    suggestions: List[str]
    confidence_score: float
    category: QueryCategory = QueryCategory.UNKNOWN
    complexity_score: float = 0.0
    estimated_cost: Optional[float] = None
    materialization_suggestions: List[str] = None
    index_suggestions: List[str] = None


class AntiPattern(BaseModel):
    """Represents a SQL antipattern with its details."""
    code: str = Field(..., description="Antipattern code (e.g., FTS001)")
    name: str = Field(..., description="Name of the antipattern")
    category: str = Field(...,
                          description="Category (PERFORMANCE, DATA_QUALITY, COMPLEXITY, BEST_PRACTICE, SECURITY, MAINTAINABILITY)")
    description: str = Field(..., description="Detailed description of the antipattern")
    impact: str = Field(..., description="Impact level (High, Medium, Low)")
    location: str = Field(..., description="Location in the query where antipattern occurs")
    suggestion: str = Field(..., description="Suggestion to fix the antipattern")


class QueryAnalysisResponse(BaseModel):
    """Structured response for query analysis."""
    antipatterns: List[AntiPattern] = Field(default_factory=list, description="List of antipatterns found in the query")
    suggestions: List[str] = Field(default_factory=list, description="List of optimization suggestions")
    complexity_score: float = Field(..., ge=0, le=1, description="Query complexity score between 0 and 1")


class QueryCategoryResponse(BaseModel):
    """Structured response for query categorization."""
    category: str = Field(..., description="Query category name")
    explanation: str = Field(..., description="Explanation for the categorization")


class QueryAnalyzer:
    """Analyzes and optimizes SQL queries using LLMs."""

    def __init__(
            self, openai_client: OpenAI,
                 openai_model: str,
                 tru_chat=None,
                 tru_app=None):
        """Initialize the analyzer with API credentials."""
        # Initialize Anthropic client
        self.client = openai_client
        self.model = openai_model

        self.chat = tru_chat
        #
        self.tru_app = tru_app

        # System message for consistent JSON responses
        self.system_message = {
            "role": "system",
            "content": """You are an expert SQL analyzer and optimizer. You must:
1. Always respond with valid JSON only
2. Never include any explanatory text outside the JSON
3. Follow the exact format shown in examples
4. Include all required fields
5. Use only the specified antipattern codes"""
        }

        # Example query and response for categorization
        self.category_example = {
            "role": "assistant",
            "content": """{
    "category": "Data Manipulation",
    "explanation": "This query retrieves and filters user records based on a date condition"
}"""
        }

        # Example query and response for analysis
        self.analysis_example = {
            "role": "assistant",
            "content": """{
    "antipatterns": [
        {
            "code": "FTS001",
            "name": "Full Table Scan",
            "category": "PERFORMANCE",
            "description": "Query uses SELECT * which retrieves unnecessary columns",
            "impact": "High",
            "location": "SELECT clause",
            "suggestion": "Specify only required columns in SELECT clause"
        }
    ],
    "suggestions": [
        "Replace SELECT * with specific column names",
        "Add index on created_at for better filtering"
    ],
    "complexity_score": 0.6
}"""
        }

        # Template for query categorization
        self.categorization_template = f"""Analyze this SQL query and categorize it.
Respond with a JSON object containing exactly these fields:
- category: One of ["Data Manipulation", "Reporting", "ETL", "Analytics"]
- explanation: Brief explanation of the categorization

Example query:
SELECT * FROM users WHERE created_at > '2024-01-01'

Example response:
{self.category_example["content"]}

Query to analyze:
{{query}}"""

        # Template for query analysis
        self.analysis_template = f"""Analyze this SQL query for antipatterns and optimization opportunities.
Respond with a JSON object containing exactly these fields:
- antipatterns: Array of antipattern objects with fields:
  - code: Antipattern code from the list below
  - name: Name of the antipattern
  - category: Category from list below
  - description: Detailed description
  - impact: "High", "Medium", or "Low"
  - location: Where in query
  - suggestion: How to fix
- suggestions: Array of optimization suggestions
- complexity_score: Number between 0 and 1

Available antipattern codes:
- PERFORMANCE: FTS001 (Full Table Scan), IJN001 (Inefficient Join), IDX001 (Missing Index), LDT001 (Large Data Transfer)
- DATA_QUALITY: NUL001 (Null Handling), DTM001 (Date/Time Manipulation)
- COMPLEXITY: NSQ001 (Nested Subquery), CJN001 (Complex Join)
- BEST_PRACTICE: WCD001 (Weak Column Definition), ALS001 (Ambiguous Column Selection)
- SECURITY: INJ001 (SQL Injection Risk), PRM001 (Permission Issues)
- MAINTAINABILITY: CMT001 (Missing Comments), FMT001 (Poor Formatting)

Example query:
SELECT * FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.created_at > '2024-01-01'

Example response:
{self.analysis_example["content"]}

Query to analyze:
{{query}}"""

    def __run_chat_completion(self, user_prompt, system_prompt=None, **chat_kwargs):
        if self.tru_app:
            with self.tru_app:
                return self.chat.chat(user_prompt, **chat_kwargs)
        else:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt}
                )
            messages.append({'role': 'user', 'content': user_prompt})

            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **chat_kwargs
            ).choices[0].message.content

    def _parse_and_validate(self, query: str, max_retries: int = 3) -> bool:
        """Validate SQL query syntax and attempt repair if needed.

        Args:
            query: SQL query string
            max_retries: Maximum number of repair attempts

        Returns:
            bool indicating if query is valid
        """
        try:
            # First attempt: Basic parsing
            parsed = parse_one(query)
            return True
        except Exception as e:
            if max_retries <= 0:
                return False

            try:
                # Clean and normalize the query
                cleaned_query = sqlparse.format(
                    query,
                    keyword_case='upper',
                    identifier_case='lower',
                    strip_comments=True,
                    reindent=True
                )

                # Handle common issues
                cleaned_query = cleaned_query.replace('\n', ' ').strip()
                cleaned_query = ' '.join(cleaned_query.split())

                # Handle temporary tables (Snowflake specific)
                if 'INTO #' in cleaned_query:
                    cleaned_query = cleaned_query.replace('INTO #', 'INTO TEMP_')

                # Handle HAVING without GROUP BY
                if 'HAVING' in cleaned_query.upper() and 'GROUP BY' not in cleaned_query.upper():
                    cleaned_query = cleaned_query.replace('HAVING', 'WHERE')

                # Handle implicit joins
                if ',' in cleaned_query and 'FROM' in cleaned_query.upper():
                    parts = cleaned_query.split('FROM')
                    if len(parts) == 2:
                        select_part = parts[0]
                        from_part = parts[1]
                        if ',' in from_part and 'JOIN' not in from_part.upper():
                            tables = [t.strip() for t in from_part.split(',')]
                            joined_tables = tables[0]
                            for i in range(1, len(tables)):
                                joined_tables += f" CROSS JOIN {tables[i]}"
                            cleaned_query = f"{select_part} FROM {joined_tables}"

                # Recursive attempt with cleaned query
                return self._parse_and_validate(cleaned_query, max_retries - 1)
            except Exception:
                return False

    def _repair_query(self, query: str, error_message: str = None) -> Optional[str]:
        """Attempt to repair an invalid SQL query using LLM."""
        repair_prompt = """You are a SQL repair expert. The following query is invalid:
{query}

Error message: {error}

Fix the syntax while preserving the same logic. Common issues to check:
1. Missing GROUP BY for aggregate functions
2. Improper HAVING clause usage
3. Invalid temporary table syntax
4. Implicit joins that should be explicit
5. Invalid function calls or data type conversions
6. Missing table aliases in complex queries
7. Invalid date/time operations

Return only valid SQL enclosed in ```sql ... ```
Do not include any other text in your response."""

        try:
            response_text = self.__run_chat_completion(
                user_prompt=repair_prompt.format(
                    query=query,
                    error=error_message or "Syntax error detected"
                ),
                system_prompt=None,
                max_tokens=2048
            )

            if "```sql" in response_text and "```" in response_text:
                sql_block = response_text.split("```sql")[1].split("```")[0].strip()

                # Validate repaired query
                if self._parse_and_validate(sql_block):
                    return sql_block

            return None
        except Exception:
            return None

    def _identify_antipatterns(self, query: str) -> List[str]:
        """Identify common SQL antipatterns.

        Args:
            query: SQL query string

        Returns:
            List of identified antipatterns
        """
        antipatterns = []
        parsed = sqlparse.parse(query)[0]

        # Check for SELECT *
        if any(token.value == '*' for token in parsed.flatten()):
            antipatterns.append("Uses SELECT * instead of specific columns")

        # Check for DISTINCT
        if 'DISTINCT' in query.upper():
            antipatterns.append("Uses DISTINCT which might indicate a join problem")

        # Check for non-SARGable conditions
        if 'LIKE' in query.upper() and '%' in query:
            antipatterns.append("Contains leading wildcard LIKE which prevents index usage")

        return antipatterns

    def _calculate_complexity_score(self, query: str) -> float:
        """Calculate a complexity score for the query.

        Args:
            query: SQL query string

        Returns:
            Float between 0 and 1 indicating query complexity
        """
        score = 0.0
        query_upper = query.upper()

        # Check for joins
        score += 0.1 * query_upper.count(" JOIN ")

        # Check for subqueries
        score += 0.2 * query_upper.count("SELECT")

        # Check for window functions
        if "OVER (" in query_upper:
            score += 0.3

        # Check for aggregations
        if any(agg in query_upper for agg in ["GROUP BY", "SUM(", "COUNT(", "AVG("]):
            score += 0.2

        # Normalize score to 0-1 range
        return min(1.0, score)

    def _suggest_indexes(
            self,
            query: str,
            schema_info: Optional[SchemaInfo] = None
    ) -> List[str]:
        """Suggest indexes based on query analysis.

        Args:
            query: SQL query string
            schema_info: Optional schema information

        Returns:
            List of index suggestions
        """
        suggestions = []
        parsed = parse_one(query)

        # Extract columns used in WHERE clauses and JOINs
        # This is a simplified version - in practice, you'd want more sophisticated analysis
        where_columns = set()
        join_columns = set()

        # Add suggestions based on findings
        if where_columns:
            suggestions.append(f"Consider adding indexes on filter columns: {', '.join(where_columns)}")
        if join_columns:
            suggestions.append(f"Consider adding indexes on join columns: {', '.join(join_columns)}")

        return suggestions

    
    def _suggest_clustering_keys(self, query: str, schema_info: Optional[SchemaInfo] = None) -> List[str]:
        """Suggest clustering keys for better query performance.

        Args:
            query: SQL query string
            schema_info: Optional schema information

        Returns:
            List of clustering key suggestions
        """
        suggestions = []
        query_upper = query.upper()
        parsed = parse_one(query)

        # Extract columns from WHERE, ORDER BY, GROUP BY clauses
        where_columns = set()
        order_columns = set()
        group_columns = set()

        try:
            # Extract table names and their filter conditions
            tables_and_filters = {}

            # Analyze query patterns
            if "GROUP BY" in query_upper and "ORDER BY" in query_upper:
                suggestions.append("Consider clustering on GROUP BY columns followed by ORDER BY columns")

            if "DATE" in query_upper or "TIMESTAMP" in query_upper:
                suggestions.append("Consider clustering on date/timestamp columns for time-based queries")

            if "PARTITION BY" in query_upper:
                suggestions.append("Align clustering keys with PARTITION BY columns for better pruning")

        except Exception:
            pass

        return suggestions

    
    def _suggest_materialized_views(self, query: str, schema_info: Optional[SchemaInfo] = None) -> List[str]:
        """Suggest materialized views for query optimization.

        Args:
            query: SQL query string
            schema_info: Optional schema information

        Returns:
            List of materialized view suggestions
        """
        suggestions = []
        query_upper = query.upper()

        # Check for expensive aggregations
        if ("GROUP BY" in query_upper and
                any(agg in query_upper for agg in ["SUM(", "COUNT(", "AVG(", "MAX(", "MIN("])):
            suggestions.append("Consider creating a materialized view for frequently used aggregations")

        # Check for complex joins
        if query_upper.count("JOIN") >= 3:
            suggestions.append("Consider materializing frequently joined tables")

        # Check for window functions
        if "OVER (" in query_upper:
            suggestions.append(
                "Consider materializing window function results if the base data changes infrequently")

        return suggestions

    
    def _suggest_search_optimization(self, query: str) -> List[str]:
        """Suggest search optimization service usage.

        Args:
            query: SQL query string

        Returns:
            List of search optimization suggestions
        """
        suggestions = []
        query_upper = query.upper()

        # Check for text search patterns
        if "LIKE" in query_upper or "CONTAINS" in query_upper:
            suggestions.append("Enable search optimization service for text search columns")

        # Check for range scans
        if "BETWEEN" in query_upper or "IN (" in query_upper:
            suggestions.append("Consider search optimization for range scan columns")

        return suggestions

    
    def _suggest_caching_strategy(self, query: str) -> List[str]:
        """Suggest query result caching strategies.

        Args:
            query: SQL query string

        Returns:
            List of caching suggestions
        """
        suggestions = []
        query_upper = query.upper()

        # Check for deterministic queries
        if not any(keyword in query_upper for keyword in ["CURRENT_TIMESTAMP", "RANDOM", "UUID_STRING"]):
            suggestions.append("Enable query result caching for deterministic queries")

        # Check for lookup patterns
        if "IN (SELECT" in query_upper or "EXISTS (SELECT" in query_upper:
            suggestions.append("Consider caching lookup table results")

        return suggestions

    def _analyze_query_structure(self, query: str) -> List[str]:
        """Analyze query structure and suggest improvements without generating code.

        Args:
            query: SQL query string

        Returns:
            List of suggested improvements
        """
        analysis_prompt = """Analyze this SQL query and suggest improvements.
    Focus on:
    1. Join optimizations
    2. Filter placement
    3. Projection optimization
    4. Snowflake-specific features
    
    Return only bullet points of suggested changes.
    Do not generate any SQL code.
    
    Query to analyze: 
    {query}
    """.lstrip()

        try:

            llm_response = self.__run_chat_completion(
                user_prompt=analysis_prompt.format(query=query),
                system_prompt=None,
                max_tokens=2048,
            )

            # Extract suggestions from response
            suggestions = []
            for line in llm_response.split('\n'):
                if line.strip().startswith('- '):
                    suggestions.append(line.strip()[2:])
            return suggestions
        except Exception:
            return []

    def _validate_schema_references(self, query: str, schema_info: Optional[SchemaInfo]) -> Tuple[
        bool, Optional[str]]:
        """Validate that query references only existing tables and columns.

        Args:
            query: SQL query string
            schema_info: Optional schema information

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not schema_info:
            return True, None

        try:
            # Parse query to extract table and column references
            parsed = parse_one(query)

            # Extract table references
            table_refs = set()
            column_refs = set()

            def visit(node):
                if isinstance(node, exp.Table):
                    table_refs.add(node.name)
                elif isinstance(node, exp.Column):
                    if node.table:
                        column_refs.add(f"{node.table}.{node.name}")
                    else:
                        column_refs.add(node.name)
                for child in node.args:
                    visit(child)

            visit(parsed)

            # Validate table references
            if schema_info.table_name not in table_refs:
                return False, f"Query references non-existent table: {table_refs - {schema_info.table_name} }"

            # Get valid column names
            valid_columns = {col["name"] for col in schema_info.columns}

            # Validate column references
            invalid_columns = set()
            for col_ref in column_refs:
                if "." in col_ref:
                    table, col = col_ref.split(".")
                    if table == schema_info.table_name and col not in valid_columns:
                        invalid_columns.add(col)
                elif col_ref not in valid_columns:
                    invalid_columns.add(col_ref)

            if invalid_columns:
                return False, f"Query references non-existent columns: {invalid_columns}"

            return True, None
        except Exception as e:
            return False, f"Schema validation failed: {str(e)}"

    
    def _generate_optimized_query(self, query: str, improvements: List[str],
                                  schema_info: Optional[SchemaInfo] = None) -> Optional[str]:
        """Generate optimized query based on suggested improvements."""
        # Add schema information to the prompt if available
        schema_context = ""
        if schema_info:
            schema_context = f"""
    Table: {schema_info.table_name}
    Columns: {', '.join(col['name'] for col in schema_info.columns)}
    """

        # Filter out infrastructure-level suggestions
        query_level_improvements = [
            imp for imp in improvements
            if not any(keyword in imp.lower() for keyword in [
                "cluster", "materiali", "cache", "index", "partition"
            ])
        ]

        # If no query-level improvements, add a default one
        if not query_level_improvements:
            query_level_improvements = [
                "Optimize query structure and performance while maintaining identical results"]

        generation_prompt = """You are an expert SQL optimizer specializing in Snowflake.
Your task is to rewrite the provided SQL query to be more efficient while maintaining identical results.

Focus on query-level optimizations such as:
1. Proper join order and type
2. Efficient filtering and predicate placement
3. Minimal projection (SELECT only needed columns)
4. Subquery optimization
5. Proper use of window functions
6. Efficient aggregation strategies

Apply these specific improvements:
{improvements}

{schema_context}

Original query:
{query}

Return only the optimized SQL query enclosed in ```sql ... ```
The query must be syntactically valid Snowflake SQL.
Only reference existing tables and columns.
Do not include any other text.
Do not add comments or explanations.
The optimized query must return exactly the same results as the original."""

        try:
            response_text = self.__run_chat_completion(
                user_prompt=generation_prompt.format(
                    query=query,
                    improvements="\n".join(f"- {imp}" for imp in query_level_improvements),
                    schema_context=schema_context
                ),
                max_tokens=2048,
            )

            # Extract and validate optimized query
            if "```sql" in response_text and "```" in response_text:
                sql_block = response_text.split("```sql")[1].split("```")[0].strip()

                # Validate optimized query with retries
                if self._parse_and_validate(sql_block, max_retries=3):
                    return sql_block
                else:
                    print("Failed to validate optimized query after retries")
                    return None

            return None
        except Exception as e:
            print(f"Error generating optimized query: {str(e)}")
            return None

    def _clean_json_response(self, response_text: str) -> str:
        """Clean and normalize JSON response from LLM."""
        try:
            # Remove all whitespace and newlines
            response_text = ''.join(response_text.split())

            # Find the first '{' and last '}'
            start = response_text.find('{')
            end = response_text.rfind('}')

            if start == -1 or end == -1:
                print(f"No valid JSON object found in response: {response_text}")
                raise ValueError("No JSON object found in response")

            # Extract just the JSON part
            response_text = response_text[start:end + 1]

            # Fix common JSON formatting issues
            response_text = response_text.replace("'", '"')
            response_text = response_text.replace('}{', '},{')
            response_text = response_text.replace('""', '"')
            response_text = response_text.replace('\\"', '"')

            # Try to parse and re-serialize to ensure valid JSON
            parsed = json.loads(response_text)
            return json.dumps(parsed, separators=(',', ':'))

        except Exception as e:
            print(f"Error cleaning JSON response: {str(e)}")
            print(f"Original response: {response_text}")
            # Return a minimal valid JSON
            return json.dumps({
                "antipatterns": [],
                "suggestions": [],
                "complexity_score": 0.5
            }, separators=(',', ':'))

    def _parse_category_response(self, response_text: str) -> dict:
        """Parse category response into structured format."""
        try:
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            result = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    result[key.strip().lower()] = value.strip()
            return {
                "category": result.get('category', 'Unknown'),
                "explanation": result.get('explanation', '')
            }
        except Exception as e:
            print(f"Error parsing category response: {str(e)}")
            return {"category": "Unknown", "explanation": ""}

    def _parse_analysis_response(self, response_text: str) -> dict:
        """Parse analysis response into structured format."""
        try:
            print("\nParsing Analysis Response:")
            print("=" * 80)
            print("Sections split by '---':")
            sections = response_text.split('---')
            print(f"Number of sections: {len(sections)}")
            for i, section in enumerate(sections):
                print(f"\nSection {i}:")
                print(section.strip())

            antipatterns = []
            suggestions = []
            complexity = 0.5

            print("\nParsing antipatterns from first section:")
            current_pattern = {}
            for line in sections[0].split('\n'):
                line = line.strip()
                if line:
                    print(f"Processing line: {line}")
                if line and ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    print(f"Found key-value pair: {key} = {value}")

                    if key == 'code' and current_pattern:
                        print(f"Completing pattern: {current_pattern}")
                        antipatterns.append(current_pattern)
                        current_pattern = {}
                    current_pattern[key] = value
            if current_pattern:
                print(f"Adding final pattern: {current_pattern}")
                antipatterns.append(current_pattern)

            print("\nParsing suggestions and complexity from remaining sections:")
            if len(sections) > 1:
                for line in sections[1].split('\n'):
                    line = line.strip()
                    print(f"Processing line: {line}")
                    if line.startswith('SUGGESTION:'):
                        suggestion = line.split(':', 1)[1].strip()
                        print(f"Found suggestion: {suggestion}")
                        suggestions.append(suggestion)
                    elif line.startswith('COMPLEXITY:'):
                        try:
                            complexity = float(line.split(':', 1)[1].strip())
                            print(f"Found complexity: {complexity}")
                        except ValueError:
                            print("Invalid complexity value")
                            complexity = 0.5

            result = {
                "antipatterns": antipatterns,
                "suggestions": suggestions,
                "complexity_score": complexity
            }
            print("\nFinal parsed result:")
            print(result)
            return result

        except Exception as e:
            print(f"\nError parsing analysis response: {str(e)}")
            print(f"Original response text:\n{response_text}")
            return {
                "antipatterns": [],
                "suggestions": [],
                "complexity_score": 0.5
            }

    
    def _get_antipatterns(self, query: str) -> List[dict]:
        """Get antipatterns using a focused prompt."""
        antipattern_prompt = f"""Analyze this SQL query for antipatterns.
For each antipattern found, provide the following information in plain text format:
CODE: (use one from the list below)
NAME: (name of the antipattern)
DESCRIPTION: (brief description)
IMPACT: (High, Medium, or Low)
LOCATION: (where in query)
SUGGESTION: (how to fix)

Available codes:
- PERFORMANCE: FTS001 (Full Table Scan), IJN001 (Inefficient Join)
- DATA_QUALITY: NUL001 (Null Handling), DTM001 (Date/Time)
- COMPLEXITY: NSQ001 (Nested Subquery), CJN001 (Complex Join)
- BEST_PRACTICE: WCD001 (Weak Columns), ALS001 (Ambiguous Columns)

Query to analyze:
{query}"""

        try:
            response = self.__run_chat_completion(
                user_prompt=antipattern_prompt,
                system_prompt="You are an expert SQL analyzer. Provide detailed analysis in a clear, structured format.",
                max_tokens=4096
            )

            # Parse the response into structured format
            antipatterns = []
            current_pattern = {}

            for line in response.split('\n'):
                line = line.strip()
                line = line.replace("*", "")
                if not line:
                    if current_pattern:
                        antipatterns.append(current_pattern)
                        current_pattern = {}
                    continue

                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()

                    if key in ['CODE', 'NAME', 'DESCRIPTION', 'IMPACT', 'LOCATION', 'SUGGESTION']:
                        current_pattern[key.lower()] = value

            if current_pattern:
                antipatterns.append(current_pattern)

            return antipatterns

        except Exception as e:
            print(f"Error getting antipatterns: {str(e)}")
            return []

    
    def _get_suggestions(self, query: str) -> List[str]:
        """Get optimization suggestions using a focused prompt."""
        suggestion_prompt = f"""Analyze this SQL query and suggest optimizations.
Provide each suggestion on a new line starting with '- '.
Focus on query structure, indexes, and Snowflake features.
Keep each suggestion brief and actionable.

Query to analyze:
{query}"""

        try:
            response = self.__run_chat_completion(
                user_prompt=suggestion_prompt,
                system_prompt="You are an expert SQL optimizer. Provide clear, actionable suggestions.",
                max_tokens=1024,
            )

            # Extract suggestions (lines starting with '- ')
            suggestions = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    suggestions.append(line[2:])
            return suggestions

        except Exception as e:
            print(f"Error getting suggestions: {str(e)}")
            return []

    
    def _get_complexity(self, query: str) -> float:
        """Get complexity score using a focused prompt."""
        complexity_prompt = f"""Analyze this SQL query's complexity.
Provide a single number between 0 and 1 representing the complexity.
Consider: joins, subqueries, window functions, aggregations.
Higher numbers indicate more complex queries.

Query to analyze:
{query}"""

        try:
            text = self.__run_chat_completion(
                user_prompt=complexity_prompt,
                system_prompt="You are an expert SQL analyzer. Provide a single number between 0 and 1.",
                max_tokens=256,
            )
            import re
            numbers = re.findall(r'0\.\d+|\d+\.?\d*', text)
            if numbers:
                score = float(numbers[0])
                return min(1.0, max(0.0, score))
            return 0.5

        except Exception as e:
            print(f"Error getting complexity: {str(e)}")
            return 0.5

    
    def _get_category(self, query: str) -> Tuple[QueryCategory, str]:
        """Get query category using a focused prompt."""
        category_prompt = f"""Analyze this SQL query and determine its category.
Provide your response in this format:
CATEGORY: (one of: Data Manipulation, Reporting, ETL, Analytics)
EXPLANATION: (brief explanation of why)

Query to analyze:
{query}"""

        try:
            response = self.__run_chat_completion(
                user_prompt=category_prompt,
                system_prompt="You are an expert SQL analyzer. Categorize the query and explain why.",
                max_tokens=1024,
            )

            # Parse the response
            category = QueryCategory.UNKNOWN
            explanation = ""

            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('CATEGORY:'):
                    cat_str = line.split(':', 1)[1].strip()
                    if cat_str in [c.value for c in QueryCategory]:
                        category = QueryCategory(cat_str)
                elif line.startswith('EXPLANATION:'):
                    explanation = line.split(':', 1)[1].strip()

            return category, explanation

        except Exception as e:
            print(f"Error getting category: {str(e)}")
            return QueryCategory.UNKNOWN, ""


    def analyze_query(
            self,
            query: str,
            schema_info: Optional[SchemaInfo] = None,
            include_cost_estimate: bool = False
    ) -> QueryAnalysis:
        """Analyze a SQL query for optimization opportunities."""
        # Initialize default values
        antipatterns = []
        suggestions = []
        complexity_score = 0.5
        category = QueryCategory.UNKNOWN
        optimized_query = None

        try:
            # Get antipatterns
            antipattern_data = self._get_antipatterns(query)
            antipatterns = [
                f"{ap.get('code', 'UNKNOWN')}: {ap.get('name', '')} - {ap.get('description', '')} (Impact: {ap.get('impact', 'Unknown')})"
                for ap in antipattern_data
            ]

            # Get suggestions
            suggestions = self._get_suggestions(query)

            # Get complexity score
            complexity_score = self._get_complexity(query)

            # Get query category
            category, category_explanation = self._get_category(query)

            # Generate optimized query with validation
            optimized_query = self._generate_optimized_query(
                query,
                suggestions if suggestions else ["Optimize query structure and performance"],
                schema_info
            )

            # Get Snowflake-specific suggestions
            try:
                clustering_suggestions = self._suggest_clustering_keys(query, schema_info)
                materialization_suggestions = self._suggest_materialized_views(query, schema_info)
                search_suggestions = self._suggest_search_optimization(query)
                caching_suggestions = self._suggest_caching_strategy(query)

                all_suggestions = (
                        suggestions +
                        clustering_suggestions +
                        materialization_suggestions +
                        search_suggestions +
                        caching_suggestions
                )
            except Exception as e:
                print(f"Error getting additional suggestions: {str(e)}")
                all_suggestions = suggestions
                materialization_suggestions = []

        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            # Fall back to basic analysis
            antipatterns = self._identify_antipatterns(query)
            all_suggestions = []
            materialization_suggestions = []

        return QueryAnalysis(
            original_query=query,
            optimized_query=optimized_query,
            antipatterns=antipatterns,
            suggestions=all_suggestions,
            confidence_score=0.8 if optimized_query else 0.5,
            category=category,
            complexity_score=complexity_score,
            estimated_cost=None,
            materialization_suggestions=materialization_suggestions,
            index_suggestions=self._suggest_indexes(query, schema_info)
        )
