"""Module for analyzing and optimizing SQL queries using LLMs."""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import sqlparse
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


class QueryAnalyzer:
    """Analyzes and optimizes SQL queries using LLMs."""

    def __init__(self, anthropic_api_key: str):
        """Initialize the analyzer with API credentials.

        Args:
            anthropic_api_key: API key for Anthropic's Claude
        """
        self.llm = ChatAnthropic(
            anthropic_api_key=anthropic_api_key,
            model="claude-3-5-sonnet-20240620"
        )
        
        # Template for query categorization
        self.categorization_template = ChatPromptTemplate.from_template(
            """You are an expert SQL query analyzer. Your task is to categorize SQL queries.
            Return ONLY a valid JSON object with this structure:
            {{"category": "CATEGORY_NAME", "explanation": "EXPLANATION"}}
            
            Valid categories are:
            - "Data Manipulation"
            - "Reporting"
            - "ETL"
            - "Analytics"
            
            Do not include any other text in your response.
            
            Query to analyze: {query}"""
        )
        
        # Template for query analysis
        self.analysis_template = ChatPromptTemplate.from_template(
            """You are an expert SQL optimizer specializing in Snowflake. 
            Analyze the provided SQL query for antipatterns and optimization opportunities.
            Focus on:
            1. Join optimizations
            2. Filter optimizations
            3. Projection optimizations
            4. Materialization opportunities
            5. Partitioning suggestions
            
            Provide specific, actionable recommendations.
            
            Query to analyze: {query}"""
        )
        
        # Template for query optimization
        self.optimization_template = ChatPromptTemplate.from_template(
            """You are an expert SQL optimizer specializing in Snowflake.
            Rewrite the provided SQL query to be more efficient while maintaining identical results.
            Focus on:
            1. Proper join order
            2. Efficient filtering
            3. Minimal projection
            4. Proper use of Snowflake features
            
            Provide the optimized query and explain your changes.
            
            Query to optimize: {query}"""
        )

    def _parse_and_validate(self, query: str) -> bool:
        """Validate SQL query syntax and attempt repair if needed.

        Args:
            query: SQL query string

        Returns:
            bool indicating if query is valid
        """
        try:
            # First attempt: Basic parsing
            parsed = parse_one(query)
            return True
        except Exception as e:
            try:
                # Clean and normalize the query
                cleaned_query = sqlparse.format(
                    query,
                    keyword_case='upper',
                    identifier_case='lower',
                    strip_comments=True,
                    reindent=True
                )
                
                # Second attempt: Parse cleaned query
                parsed = parse_one(cleaned_query)
                return True
            except Exception as parse_error:
                return False

    def _repair_query(self, query: str, error_message: str = None) -> Optional[str]:
        """Attempt to repair an invalid SQL query using LLM.
        
        Args:
            query: The invalid SQL query
            error_message: Optional error message from parser
            
        Returns:
            Repaired query if successful, None otherwise
        """
        repair_prompt = ChatPromptTemplate.from_template(
            """You are a SQL repair expert. The following query is invalid:
            {query}
            
            Error message: {error}
            
            Please fix the syntax while preserving the same logic.
            Return only valid SQL enclosed in ```sql ... ```
            Do not include any other text in your response."""
        )
        
        try:
            repair_response = self.llm.invoke(
                repair_prompt.format_messages(
                    query=query,
                    error=error_message or "Syntax error detected"
                )
            )
            
            # Extract SQL from response
            response_text = repair_response.content
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
            suggestions.append("Consider materializing window function results if the base data changes infrequently")
            
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
        analysis_prompt = ChatPromptTemplate.from_template(
            """Analyze this SQL query and suggest improvements.
            Focus on:
            1. Join optimizations
            2. Filter placement
            3. Projection optimization
            4. Snowflake-specific features
            
            Return only bullet points of suggested changes.
            Do not generate any SQL code.
            
            Query to analyze: {query}"""
        )
        
        try:
            analysis_response = self.llm.invoke(
                analysis_prompt.format_messages(query=query)
            )
            
            # Extract suggestions from response
            suggestions = []
            for line in analysis_response.content.split('\n'):
                if line.strip().startswith('- '):
                    suggestions.append(line.strip()[2:])
            return suggestions
        except Exception:
            return []

    def _validate_schema_references(self, query: str, schema_info: Optional[SchemaInfo]) -> Tuple[bool, Optional[str]]:
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
                return False, f"Query references non-existent table: {table_refs - {schema_info.table_name}}"
            
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

    def _generate_optimized_query(self, query: str, improvements: List[str], schema_info: Optional[SchemaInfo] = None) -> Optional[str]:
        """Generate optimized query based on suggested improvements.
        
        Args:
            query: Original SQL query
            improvements: List of suggested improvements
            schema_info: Optional schema information
            
        Returns:
            Optimized query if successful, None otherwise
        """
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
            query_level_improvements = ["Optimize query structure and performance while maintaining identical results"]
        
        generation_prompt = ChatPromptTemplate.from_template(
            """You are an expert SQL optimizer specializing in Snowflake.
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
        )

        try:
            generation_response = self.llm.invoke(
                generation_prompt.format_messages(
                    query=query,
                    improvements="\n".join(f"- {imp}" for imp in query_level_improvements),
                    schema_context=schema_context
                )
            )
            
            # Extract and validate optimized query
            response_text = generation_response.content
            if "```sql" in response_text and "```" in response_text:
                sql_block = response_text.split("```sql")[1].split("```")[0].strip()
                
                # Validate syntax
                if not self._parse_and_validate(sql_block):
                    return None
                    
                # Validate schema references
                is_valid, error = self._validate_schema_references(sql_block, schema_info)
                if not is_valid:
                    # Try repair with schema error
                    repaired = self._repair_query(sql_block, error)
                    if repaired and self._parse_and_validate(repaired):
                        is_valid, error = self._validate_schema_references(repaired, schema_info)
                        if is_valid:
                            return repaired
                    return None
                    
                return sql_block
                
            return None
        except Exception:
            return None

    def analyze_query(
        self,
        query: str,
        schema_info: Optional[SchemaInfo] = None,
        include_cost_estimate: bool = False
    ) -> QueryAnalysis:
        """Analyze a SQL query for optimization opportunities.

        Args:
            query: SQL query string
            schema_info: Optional schema information
            include_cost_estimate: Whether to include cost estimation

        Returns:
            QueryAnalysis object containing analysis results
        
        Raises:
            ValueError: If query syntax is invalid and cannot be repaired
        """
        # Validate query syntax with repair attempts
        if not self._parse_and_validate(query):
            repaired_query = self._repair_query(query)
            if repaired_query:
                query = repaired_query
            else:
                raise ValueError("Invalid SQL query syntax and repair failed")
        
        # Validate schema references in original query
        if schema_info:
            is_valid, error = self._validate_schema_references(query, schema_info)
            if not is_valid:
                raise ValueError(f"Invalid schema references in query: {error}")
        
        # Two-step optimization process
        # Step 1: Analyze and suggest improvements
        suggested_improvements = self._analyze_query_structure(query)
        
        # Step 2: Generate optimized query based on improvements
        # Always attempt to generate an optimized query
        optimized_query = self._generate_optimized_query(
            query,
            suggested_improvements if suggested_improvements else ["Optimize query structure and performance"],
            schema_info
        )
        
        # Get other analysis components
        antipatterns = self._identify_antipatterns(query)
        complexity_score = self._calculate_complexity_score(query)
        
        # Get query category
        category_response = self.llm.invoke(
            self.categorization_template.format_messages(query=query)
        )
        
        try:
            response_text = category_response.content.strip()
            category_data = json.loads(response_text)
            category_name = category_data.get("category", "Unknown")
            if category_name in [c.value for c in QueryCategory]:
                category = QueryCategory(category_name)
            else:
                category = QueryCategory.UNKNOWN
        except (json.JSONDecodeError, ValueError, KeyError):
            category = QueryCategory.UNKNOWN
        
        # Get Snowflake-specific suggestions
        clustering_suggestions = self._suggest_clustering_keys(query, schema_info)
        materialization_suggestions = self._suggest_materialized_views(query, schema_info)
        search_suggestions = self._suggest_search_optimization(query)
        caching_suggestions = self._suggest_caching_strategy(query)
        
        # Combine all suggestions
        all_suggestions = (
            suggested_improvements +
            clustering_suggestions +
            materialization_suggestions +
            search_suggestions +
            caching_suggestions
        )
        
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