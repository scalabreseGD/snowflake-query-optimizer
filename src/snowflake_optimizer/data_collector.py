"""Module for collecting query performance data from Snowflake."""
import json
from typing import Dict, Any, Optional, Literal
import collections

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

    @streamlit.cache_data(show_spinner="Fetching Query History")
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
    
    @streamlit.cache_data(show_spinner=False)
    def get_impacted_objects(_self, query_id:str) -> Dict[str, Any]:
        """
        Fetch impacted objects metadata by query_id

         Args:
            query_id: The Snowflake query ID

        Returns:
            Dictionary containing the objects metadata
        """
        query = f"""
            select
                boa.value:"objectName"::string as table_name
            from snowflake.account_usage.access_history
            , lateral flatten(base_objects_accessed) boa
            where query_id = '{query_id}'
            and  boa.value:"objectDomain"::string='Table';"""
        with _self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    
    @streamlit.cache_data(show_spinner=False)
    def get_impacted_objects_metadata(_self, impacted_objects: pd.DataFrame) -> Dict[str, Any]:
        """
        Fetch impacted objects metadata by query_id

         Args:
            impacted_objects: The Snowflake impacted objects

        Returns:
            Dictionary containing the objects metadata
        """
        metadata = collections.defaultdict(dict)
        for _,row in impacted_objects.iterrows():
            object_name = row.iloc[0]
            table_catalog, table_schema, table_name = object_name.split('.')
            # desc_query = f"""DESC TABLE {object_name}"""
            desc_query = f"""
                SELECT OBJECT_AGG(COLUMN_NAME, DATA_TYPE::VARIANT) as table_schema FROM SNOWFLAKE.ACCOUNT_USAGE.COLUMNS 
                WHERE 
                TABLE_CATALOG = '{table_catalog}' AND
                TABLE_SCHEMA = '{table_schema}' AND
                TABLE_NAME = '{table_name}'
                AND DELETED IS NULL;
                """
            try:
                with _self.engine.connect() as conn:
                    desc_dict = pd.read_sql(desc_query, conn).to_dict(orient="records") 
                    metadata[object_name]["table_schema"] = desc_dict
            except:
                print(f"No metadata for object: {object_name} in SNOWFLAKE.ACCOUNT_USAGE.COLUMNS")
            try:   
                table_catalog, table_schema, table_name = object_name.split('.')
                query = f"""
                    SELECT ROW_COUNT, BYTES, CLUSTERING_KEY, AUTO_CLUSTERING_ON, TABLE_TYPE
                    FROM SNOWFLAKE.ACCOUNT_USAGE.TABLES 
                    WHERE
                    table_name = '{table_name}' AND
                    table_schema = '{table_schema}' AND
                    table_catalog = '{table_catalog}';
                    """
                with _self.engine.connect() as conn:
                    metadata_dict = pd.read_sql(query, conn).to_dict(orient="records")
                    metadata[object_name]["metadata"] = metadata_dict 
            except:
                print(f"No metadata for object: {object_name} in SNOWFLAKE.ACCOUNT_USAGE.TABLES")
        return dict(metadata)

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

    def get_query_operator_stats_by_query_id(self, query_id: str):
        query = f"SELECT TO_JSON(*) as OPERATOR_STATS FROM (select ARRAY_AGG(OBJECT_CONSTRUCT(*)) from TABLE(GET_QUERY_OPERATOR_STATS('{query_id}')))"
        res = self.snowpark_session.sql(query).collect()
        operator_stats = res[0]['OPERATOR_STATS']
        return operator_stats

    def compare_optimized_query_with_original(self, optimized_query, original_query_id):
        with SnowflakeTransaction(session=self.snowpark_session, action_on_complete='rollback'):
            self.snowpark_session.sql(optimized_query).to_pandas()
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
