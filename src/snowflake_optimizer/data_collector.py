"""Module for collecting query performance data from Snowflake."""

from typing import Dict, Any, Optional, Literal

import pandas as pd
import streamlit
from snowflake.snowpark import Session
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
        self.engine = create_engine(
            'snowflake://{user}:{password}@{account}/'.format(
                **self.connection_params
            )
        )

        self.snowpark_session = Session.builder.configs(self.connection_params).create()

    def get_expensive_queries_paginated(self, days: int = 7,
                                        min_execution_time: float = 60.0,
                                        limit: int = 100,
                                        page=0, page_size=20):
        df = self.get_expensive_queries(days, min_execution_time, limit)
        start_idx = page * page_size
        end_idx = start_idx + page_size

        total_pages = (len(df) - 1) // page_size + 1
        return df.iloc[start_idx:end_idx], total_pages

    @streamlit.cache_data
    def get_expensive_queries(
            _self,
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
        query = f"""WITH RankedQueries AS (
            SELECT 
                QUERY_ID,
                QUERY_TEXT,
                QUERY_HASH,
                USER_NAME,
                DATABASE_NAME,
                SCHEMA_NAME,
                WAREHOUSE_NAME,
                TOTAL_ELAPSED_TIME / 1000 AS EXECUTION_TIME_SECONDS,
                BYTES_SCANNED / 1024 / 1024 AS MB_SCANNED,
                ROWS_PRODUCED,
                COMPILATION_TIME / 1000 AS COMPILATION_TIME_SECONDS,
                EXECUTION_STATUS,
                ERROR_MESSAGE,
                START_TIME,
                END_TIME,
                PARTITIONS_SCANNED,
                PARTITIONS_TOTAL,
                BYTES_SPILLED_TO_LOCAL_STORAGE / 1024 / 1024 AS MB_SPILLED_TO_LOCAL,
                BYTES_SPILLED_TO_REMOTE_STORAGE / 1024 / 1024 AS MB_SPILLED_TO_REMOTE,
                ROW_NUMBER() OVER (PARTITION BY QUERY_HASH ORDER BY START_TIME DESC) AS RN
            FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
            WHERE 
                TOTAL_ELAPSED_TIME >= {min_execution_time * 1000}
                AND DATABASE_NAME = 'IDS_PROD'
                AND START_TIME >= DATEADD('days', -{days}, CURRENT_TIMESTAMP())
                AND EXECUTION_STATUS = 'SUCCESS'
                AND QUERY_TYPE = 'SELECT'
        )
        SELECT *
        FROM RankedQueries
        WHERE RN = 1
        ORDER BY EXECUTION_TIME_SECONDS DESC
        LIMIT {limit};"""

        with _self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn
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

    def compare_optimized_query_with_original(self, optimized_query, original_query_id):
        with SnowflakeTransaction(session=self.snowpark_session, action_on_complete='rollback'):
            self.snowpark_session.sql(optimized_query).collect()
            res = self.snowpark_session.sql("SELECT LAST_QUERY_ID()").collect()
            print()


class SnowflakeTransaction:
    def __init__(
            self,
            session: Session,
            action_on_complete: Literal["commit", "rollback"],
            action_on_error: Literal["commit", "rollback"] = 'rollback',
    ):
        self.session = session
        self.actioned = False
        self.action_on_complete = action_on_complete
        if action_on_complete not in ["commit", "rollback"]:
            raise ValueError(f"Invalid action_on_complete action {action_on_complete}")
        self.action_on_error = action_on_error
        if action_on_error not in ["commit", "rollback"]:
            raise ValueError(f"Invalid action_on_error action {action_on_error}")

    def __enter__(self):
        self.session.sql("begin transaction").collect()
        return self

    def commit(self):
        self.session.sql("commit").collect()
        self.actioned = True

    def rollback(self):
        self.session.sql("rollback").collect()
        self.actioned = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if we already actioned, don't do anything
        if not self.actioned:
            # if an error was thrown, rollback
            if exc_type is not None:
                self.session.sql(self.action_on_error).collect()
            else:
                self.session.sql(self.action_on_complete).collect()
