"""Module for collecting query performance data from Snowflake."""

from typing import List, Dict, Any, Optional
import pandas as pd
from snowflake.connector import SnowflakeConnection
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine


class QueryMetricsCollector:
    """Collects query performance metrics from Snowflake."""

    def __init__(
        self,
        account: str,
        user: str,
        password: str,
        warehouse: str,
        database: Optional[str] = None,
        schema: Optional[str] = None,
    ):
        """Initialize the collector with Snowflake credentials.

        Args:
            account: Snowflake account identifier
            user: Snowflake username
            password: Snowflake password
            warehouse: Snowflake warehouse name
            database: Optional database name
            schema: Optional schema name
        """
        self.connection_params = {
            "account": account,
            "user": user,
            "password": password,
            "warehouse": warehouse,
            "database": database,
            "schema": schema,
        }
        self.engine = create_engine(URL(**self.connection_params))

    def get_expensive_queries(
        self,
        days: int = 7,
        min_execution_time: float = 60.0,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch expensive queries from Snowflake query history.

        Args:
            days: Number of days to look back
            min_execution_time: Minimum execution time in seconds
            limit: Maximum number of queries to return

        Returns:
            DataFrame containing query metrics
        """
        query = """
        SELECT 
            QUERY_ID,
            QUERY_TEXT,
            DATABASE_NAME,
            SCHEMA_NAME,
            WAREHOUSE_NAME,
            TOTAL_ELAPSED_TIME/1000 as EXECUTION_TIME_SECONDS,
            BYTES_SCANNED/1024/1024 as MB_SCANNED,
            ROWS_PRODUCED,
            COMPILATION_TIME/1000 as COMPILATION_TIME_SECONDS,
            EXECUTION_STATUS,
            ERROR_MESSAGE,
            START_TIME,
            END_TIME,
            TOTAL_PARTITIONS_SCANNED,
            PARTITIONS_TOTAL,
            BYTES_SPILLED_TO_LOCAL_STORAGE/1024/1024 as MB_SPILLED_TO_LOCAL,
            BYTES_SPILLED_TO_REMOTE_STORAGE/1024/1024 as MB_SPILLED_TO_REMOTE
        FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY(
            DATE_RANGE_START=>DATEADD('days', -:1, CURRENT_TIMESTAMP())
        ))
        WHERE TOTAL_ELAPSED_TIME >= :2
        AND EXECUTION_STATUS = 'SUCCESS'
        AND QUERY_TYPE = 'SELECT'
        ORDER BY TOTAL_ELAPSED_TIME DESC
        LIMIT :3
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params=[days, min_execution_time * 1000, limit]
            )
        return df

    def get_query_plan(self, query_id: str) -> Dict[str, Any]:
        """Fetch the query execution plan for a specific query.

        Args:
            query_id: The Snowflake query ID

        Returns:
            Dictionary containing the query plan details
        """
        query = "SELECT * FROM TABLE(GET_QUERY_OPERATOR_STATS(:1))"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params=[query_id])
        return df.to_dict(orient="records") 