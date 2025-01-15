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
        """Validate SQL query syntax.

        Args:
            query: SQL query string

        Returns:
            bool indicating if query is valid
        """
        try:
            parsed = parse_one(query)
            return True
        except Exception:
            return False

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
        """
        # Validate query syntax
        if not self._parse_and_validate(query):
            raise ValueError("Invalid SQL query syntax")
            
        # Identify basic antipatterns
        antipatterns = self._identify_antipatterns(query)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(query)
        
        # Get query category
        category_response = self.llm.invoke(
            self.categorization_template.format_messages(query=query)
        )
        
        try:
            # Clean and parse the response
            response_text = category_response.content.strip()
            category_data = json.loads(response_text)
            category_name = category_data.get("category", "Unknown")
            # Ensure exact match with enum values
            if category_name in [c.value for c in QueryCategory]:
                category = QueryCategory(category_name)
            else:
                category = QueryCategory.UNKNOWN
        except (json.JSONDecodeError, ValueError, KeyError):
            category = QueryCategory.UNKNOWN
        
        # Get advanced Snowflake-specific suggestions
        clustering_suggestions = self._suggest_clustering_keys(query, schema_info)
        materialization_suggestions = self._suggest_materialized_views(query, schema_info)
        search_suggestions = self._suggest_search_optimization(query)
        caching_suggestions = self._suggest_caching_strategy(query)
        
        # Get LLM analysis with enhanced context
        analysis_template_with_context = self.analysis_template.format_messages(
            query=query,
            context=f"""Consider Snowflake-specific features:
            - Clustering keys
            - Materialized views
            - Search optimization
            - Query result caching
            - Micro-partitions
            - Zero-copy cloning
            - Time travel
            """
        )
        analysis_response = self.llm.invoke(analysis_template_with_context)
        
        # Get optimized query with Snowflake-specific optimizations
        optimization_template_with_context = self.optimization_template.format_messages(
            query=query,
            context="""Use Snowflake-specific features:
            - CLUSTER BY for optimal data organization
            - Materialized views for complex aggregations
            - Search optimization for text searches
            - Result cache for deterministic queries
            - Micro-partitioning for efficient pruning
            """
        )
        optimization_response = self.llm.invoke(optimization_template_with_context)
        
        # Extract optimized query and suggestions
        suggestions = []
        optimized_query = None
        confidence_score = 0.8
        
        # Parse LLM responses and extract suggestions
        for line in analysis_response.content.split('\n'):
            if line.strip().startswith('- '):
                suggestions.append(line.strip()[2:])
        
        # Add advanced Snowflake-specific suggestions
        suggestions.extend(clustering_suggestions)
        suggestions.extend(materialization_suggestions)
        suggestions.extend(search_suggestions)
        suggestions.extend(caching_suggestions)
        
        # Extract optimized query from the response
        response_lines = optimization_response.content.split('\n')
        in_sql_block = False
        sql_lines = []
        
        for line in response_lines:
            if '```sql' in line:
                in_sql_block = True
                continue
            elif '```' in line and in_sql_block:
                in_sql_block = False
                break
            elif in_sql_block:
                sql_lines.append(line)
        
        if sql_lines:
            optimized_query = '\n'.join(sql_lines).strip()
            # Validate optimized query
            if not self._parse_and_validate(optimized_query):
                optimized_query = None
        
        # Get index suggestions if schema info is provided
        index_suggestions = self._suggest_indexes(query, schema_info) if schema_info else []
        
        return QueryAnalysis(
            original_query=query,
            optimized_query=optimized_query,
            antipatterns=antipatterns,
            suggestions=suggestions,
            confidence_score=confidence_score,
            category=category,
            complexity_score=complexity_score,
            estimated_cost=None,
            materialization_suggestions=materialization_suggestions,
            index_suggestions=index_suggestions
        ) 