# data_functions.py
import os
import psycopg2
from psycopg2 import pool
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from compare_tumor.constant import *
import logging
import psycopg2
import pandas as pd
from contextlib import contextmanager
import functools
import time
from typing import List, Dict, Optional, Tuple, Union
import psutil
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Adjust to DEBUG for more verbose logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("app.log")  # Save logs to a file
    ]
)

all_columns = []

region = ["cecum", "ascending", "transverse",
          "descending", "sigmoid", "rectosigmoid", "rectum"]

load_dotenv()
db_url = os.getenv('DATABASE_URL')

import time
import logging

def log_time(label):
    """
    Enhanced time logging decorator with more detailed performance tracking
    
    Args:
        label (str): A descriptive label for the function being timed
    
    Returns:
        Decorator function that logs execution time and additional metrics
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Capture start time and initial memory
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Log input arguments for context (safely)
            input_info = {}
            try:
                for i, arg in enumerate(args):
                    if hasattr(arg, '__len__'):
                        input_info[f'arg_{i}'] = len(arg)
                for k, v in kwargs.items():
                    if hasattr(v, '__len__'):
                        input_info[k] = len(v)
            except Exception:
                input_info = "Unable to capture input sizes"
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Calculate performance metrics
                execution_time = time.time() - start_time
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = end_memory - start_memory
                
                # Detailed logging
                logging.info(
                    f"[PERFORMANCE] {label}\n"
                    f"  Execution Time: {execution_time:.4f} seconds\n"
                    f"  Memory Used: {memory_used:.2f} MB\n"
                    f"  Input Sizes: {input_info}"
                )
                
                return result
            
            except Exception as e:
                # Log error with timing information
                execution_time = time.time() - start_time
                logging.error(
                    f"[PERFORMANCE] {label} - FAILED\n"
                    f"  Execution Time: {execution_time:.4f} seconds\n"
                    f"  Error: {str(e)}\n"
                    f"  Input Sizes: {input_info}"
                )
                raise
        
        return wrapper
    return decorator

def memory_logger(label):
    """Decorator to log memory usage before and after function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get memory before function execution
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            logging.info(f"[MEMORY] {label} - BEFORE: {memory_before:.2f} MB")
            
            try:
                result = func(*args, **kwargs)
                
                # Get memory after function execution
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = memory_after - memory_before
                logging.info(f"[MEMORY] {label} - AFTER: {memory_after:.2f} MB (DIFF: {memory_diff:+.2f} MB)")
                
                # Force garbage collection if memory increased significantly
                if memory_diff > 50:  # If memory increased by more than 50MB
                    logging.warning(f"[MEMORY] {label} - Large memory increase detected, forcing garbage collection")
                    gc.collect()
                    memory_after_gc = process.memory_info().rss / 1024 / 1024  # MB
                    logging.info(f"[MEMORY] {label} - AFTER GC: {memory_after_gc:.2f} MB")
                
                return result
            except Exception as e:
                # Log memory even if function fails
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = memory_after - memory_before
                logging.error(f"[MEMORY] {label} - ERROR: {memory_after:.2f} MB (DIFF: {memory_diff:+.2f} MB)")
                raise
        return wrapper
    return decorator

def log_memory_usage(label=""):
    """Utility function to log current memory usage"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"[MEMORY] {label} - Current: {memory_mb:.2f} MB")

# ===== CONNECTION POOL SETUP =====
class DatabasePool:
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabasePool, cls).__new__(cls)
        return cls._instance
    
    def initialize_pool(self, minconn=2, maxconn=10):
        """Initialize the connection pool"""
        try:
            if self._pool is None:
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn, maxconn, db_url
                )
                logging.info(f"Database connection pool initialized with {minconn}-{maxconn} connections")
        except Exception as e:
            logging.error(f"Error initializing connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool"""
        try:
            if self._pool is None:
                self.initialize_pool()
            return self._pool.getconn()
        except Exception as e:
            logging.error(f"Error getting connection from pool: {e}")
            raise
    
    def put_connection(self, connection):
        """Return a connection to the pool"""
        try:
            if self._pool and connection:
                self._pool.putconn(connection)
        except Exception as e:
            logging.error(f"Error returning connection to pool: {e}")
    
    def close_all(self):
        """Close all connections in the pool"""
        try:
            if self._pool:
                self._pool.closeall()
                self._pool = None
                logging.info("All database connections closed")
        except Exception as e:
            logging.error(f"Error closing connection pool: {e}")

# Initialize the pool singleton
db_pool = DatabasePool()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    connection = None
    cursor = None
    try:
        connection = db_pool.get_connection()
        cursor = connection.cursor()
        yield cursor
    except Exception as e:
        if connection:
            connection.rollback()
        logging.error(f"Database error: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if connection:
            db_pool.put_connection(connection)

# ===== CACHING DECORATORS =====
def simple_cache(max_size=128, ttl=300):  # 5 minute TTL
    """Simple in-memory cache decorator"""
    def decorator(func):
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            current_time = time.time()
            
            # Check if cached result is valid
            if key in cache and key in cache_times:
                if current_time - cache_times[key] < ttl:
                    logging.debug(f"Cache hit for {func.__name__}")
                    return cache[key]
                else:
                    # Remove expired entry
                    del cache[key]
                    del cache_times[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Manage cache size
            if len(cache) >= max_size:
                # Remove oldest entry
                oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            cache[key] = result
            cache_times[key] = current_time
            logging.debug(f"Cache miss for {func.__name__} - result stored")
            return result
        
        # Add cache management methods
        wrapper.clear_cache = lambda: cache.clear() or cache_times.clear()
        wrapper.cache_info = lambda: {"size": len(cache), "max_size": max_size, "ttl": ttl}
        
        return wrapper
    return decorator

def selected_mz_cleaning(selected_mz):
    if "'" in selected_mz:
        selected_mz = selected_mz.replace("'", "''")
        # print("updated mz value", selected_mz)
    return selected_mz

@memory_logger("get_gmm_name: Memory")
@log_time("get_gmm_name: DB Query")
@simple_cache(max_size=50, ttl=600)
def get_gmm_name(table_name):
    """
    Fetches all distinct values from the 'name' column in the specified table.
    
    Args:
        table_name (str): Name of the table in the database.
        
    Returns:
        list: Sorted list of distinct name values.
    """
    try:
        with get_db_connection() as cursor:
            # Check if table exists first
            check_table_query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
            """
            cursor.execute(check_table_query, (table_name,))
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logging.error(f"Table '{table_name}' does not exist in the database.")
                return []
                
            # Get distinct name values
            query_gmm_name = f"SELECT DISTINCT name FROM {table_name} ORDER BY name"
            cursor.execute(query_gmm_name)
            mz_values = [row[0] for row in cursor.fetchall()]
            
            # Sort the values
            mz_values = sorted(mz_values, key=lambda s: str(s).casefold() if isinstance(s, str) else s)
            logging.info(f"Retrieved {len(mz_values)} distinct name values from {table_name}")
            
            return mz_values
            
    except Exception as e:
        logging.error(f"Error retrieving name values from {table_name}: {e}")
        return []




@memory_logger("get_all_columns_data: Memory")
@log_time("get_all_columns_data: DB Query")
def get_all_columns_data(table_name, selected_compound):
    """
    Fetches all rows from a specific table for the selected compound.

    Args:
        table_name (str): Name of the table in the database.
        selected_compound (str): Name of the compound to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame containing all rows for the selected compound.
    """
    logging.info("Fetching data from table: %s for compound: %s", table_name, selected_compound)

    try:
        with get_db_connection() as cursor:
            # Fetch column names
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
            columns = [desc[0] for desc in cursor.description]
            logging.info("Columns fetched: %s", columns)

            # Ensure 'name' column exists
            if 'name' not in columns:
                logging.error("'name' column not found in table: %s", table_name)
                return None

            # Fetch all rows for the selected compound
            query = f"SELECT * FROM {table_name} WHERE name = %s"
            cursor.execute(query, (selected_compound,))
            data = cursor.fetchall()

            if not data:
                logging.warning("No data found for compound: %s", selected_compound)
                return None

            logging.info("Data fetched successfully for compound: %s", selected_compound)

            # Create DataFrame
            df = pd.DataFrame(data, columns=columns)
            columns_to_exclude = ['mz', 'rt', 'list_2_match']
            df = df.drop(columns=columns_to_exclude, errors='ignore')
            logging.info("DataFrame created successfully for compound: %s", selected_compound)
            
            # Remove duplicate rows
            df = df.drop_duplicates()
            logging.info("Duplicate rows removed, resulting DataFrame shape: %s", df.shape)

            return df

    except Exception as e:
        logging.error("Error fetching data for compound %s in table %s: %s", selected_compound, table_name, e)
        return None


@memory_logger("get_all_columns_data_all_compounds: Memory")
@log_time("get_all_columns_data_all_compounds: DB Query")
@simple_cache(max_size=10, ttl=900)  # Cache for 15 minutes - larger tables need longer cache
def get_all_columns_data_all_compounds(table_name):
    """
    Fetches all rows and columns from the given table.
    This is an expensive operation and results are cached.

    Args:
        table_name (str): Name of the table in the database.

    Returns:
        pd.DataFrame: DataFrame containing all rows and columns.
    """
    logging.info("Fetching data from table: %s", table_name)
    print(f"Fetching data from table: {table_name}")  # Debug log

    try:
        with get_db_connection() as cursor:
            # Fetch all data
            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)
            data = cursor.fetchall()

            # Get column names
            columns = [desc[0] for desc in cursor.description]
            logging.info("Columns fetched: %s", columns)

            # Create DataFrame
            df = pd.DataFrame(data, columns=columns)
            columns_to_exclude = ['mz', 'rt', 'list_2_match']
            df = df.drop(columns=columns_to_exclude, errors='ignore')
            logging.info("DataFrame created successfully for the table: %s", table_name)

            # Drop duplicates if any
            df = df.drop_duplicates()
            print(f"DataFrame shape after processing: {df.shape}")  # More efficient debug log

            return df

    except Exception as e:
        logging.error("Error fetching data: %s", e)
        print(f"Error fetching data: {e}")  # Debug log
        return None



@memory_logger("get_multiple_bacteria_top_metabolites: Memory")
@log_time("get_multiple_bacteria_top_metabolites: DB Query")
@simple_cache(max_size=50, ttl=600)
def get_multiple_bacteria_top_metabolites(table_name, selected_bacteria):
    """
    OPTIMIZED: Fetches metabolites for selected bacteria where they rank in top 10 producers.
    Maintains original logic: ranks selected bacteria against ALL bacteria, not just selected bacteria.
    
    Args:
        table_name (str): Name of the table in the database.
        selected_bacteria (list): List of bacteria to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame containing metabolites for selected bacteria in the top 10.
    """
    logging.info("Fetching top metabolites for selected bacteria: %s", selected_bacteria)
    if not selected_bacteria:
        return None

    try:
        with get_db_connection() as cursor:
            # Validate that selected bacteria columns exist
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND column_name = ANY(%s)
            """, (table_name, selected_bacteria))
            
            valid_bacteria = [row[0] for row in cursor.fetchall()]
            if not valid_bacteria:
                logging.warning("None of the selected bacteria columns exist in table: %s", table_name)
                return None
            
            logging.info("Valid bacteria columns: %s", valid_bacteria)
            
            # Get ALL bacteria columns for proper ranking
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND data_type IN ('double precision', 'numeric', 'integer', 'real', 'float', 'decimal')
                AND column_name NOT IN ('name', 'mz', 'rt', 'list_2_match', 'Type', 'Metabolite')
            """, (table_name,))
            all_bacteria_columns = [row[0] for row in cursor.fetchall()]
            
            logging.info("Total bacteria columns for ranking: %d", len(all_bacteria_columns))
            
            # HIGHLY OPTIMIZED QUERY: Use efficient ranking with minimal data processing
            # Build the ranking query more efficiently
            union_parts = []
            for col in all_bacteria_columns:
                union_parts.append(f'SELECT name, \'{col}\' AS bacteria, "{col}"::double precision AS value FROM {table_name} WHERE "{col}" IS NOT NULL AND "{col}" > 0')
            union_query = ' UNION ALL '.join(union_parts)
            
            ranking_query = f"""
            WITH AllBacteriaRanking AS (
                SELECT 
                    name AS metabolite,
                    bacteria::text,
                    value::double precision,
                    ROW_NUMBER() OVER (PARTITION BY name ORDER BY value DESC) AS rank
                FROM (
                    {union_query}
                ) all_data
            ),
            Top10PerMetabolite AS (
                SELECT metabolite, bacteria, value, rank
                FROM AllBacteriaRanking 
                WHERE rank <= 10
            )
            SELECT metabolite, bacteria, value, rank
            FROM Top10PerMetabolite
            WHERE bacteria = ANY(%s)
            ORDER BY metabolite, rank;
            """
            
            logging.info("Executing optimized query with proper ranking against all %d bacteria", len(all_bacteria_columns))
            
            cursor.execute(ranking_query, (valid_bacteria,))
            data = cursor.fetchall()
            
            if not data:
                logging.warning("No data found for selected bacteria in the top 10 metabolites: %s", selected_bacteria)
                return None

            # Create DataFrame with optimized memory usage
            columns = ['metabolite', 'bacteria', 'value', 'rank']
            df = pd.DataFrame(data, columns=columns)
            
            # Convert value to numeric efficiently
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            
            logging.info("Data fetched successfully. Shape: %s", df.shape)
            return df

    except Exception as e:
        logging.error("Error fetching data: %s", e)
        return None

 

@memory_logger("get_multiple_bacteria_cumm_top_metabolites: Memory")
@log_time("get_multiple_bacteria_cumm_top_metabolites: DB Query")
@simple_cache(max_size=50, ttl=600)
def get_multiple_bacteria_cumm_top_metabolites(table_name, selected_bacteria):
    """
    HIGHLY OPTIMIZED: Fetches metabolites where ALL selected bacteria appear together in the top 10 producers.
    Uses efficient aggregation and ranking to avoid massive UNION ALL operations.
    Maintains original logic: ranks selected bacteria against ALL bacteria, not just selected bacteria.
    """
    logging.info("Fetching cumulative top metabolites for selected bacteria: %s", selected_bacteria)
    if not selected_bacteria:
        return None

    try:
        with get_db_connection() as cursor:
            # Validate that selected bacteria columns exist
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND column_name = ANY(%s)
            """, (table_name, selected_bacteria))
            
            valid_bacteria = [row[0] for row in cursor.fetchall()]
            if not valid_bacteria:
                logging.warning("None of the selected bacteria columns exist in table: %s", table_name)
                return None
            
            logging.info("Valid bacteria columns for cumulative analysis: %s", valid_bacteria)
            
            # Get ALL bacteria columns for proper ranking
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND data_type IN ('double precision', 'numeric', 'integer', 'real', 'float', 'decimal')
                AND column_name NOT IN ('name', 'mz', 'rt', 'list_2_match', 'Type', 'Metabolite')
            """, (table_name,))
            all_bacteria_columns = [row[0] for row in cursor.fetchall()]
            
            logging.info("Total bacteria columns for cumulative ranking: %d", len(all_bacteria_columns))
            
            # Build the selected bacteria columns for the main query
            selected_cols = [f'"{col}"' for col in valid_bacteria]
            selected_cols_str = ', '.join(selected_cols)
            
            # Build the ranking comparison columns (all bacteria for proper ranking)
            ranking_cols = [f'"{col}"' for col in all_bacteria_columns]
            ranking_cols_str = ', '.join(ranking_cols)
            
            # HIGHLY OPTIMIZED QUERY: Use efficient ranking with minimal data processing
            # First, get metabolites where ALL selected bacteria have values > 0
            values_clause = ', '.join([f'("{col}", "{col}")' for col in valid_bacteria])
            cross_join_values = ', '.join([f'("{col}", t_data."{col}")' for col in valid_bacteria])
            union_parts = []
            for col in all_bacteria_columns:
                union_parts.append(f'SELECT name, \'{col}\' AS bacteria, "{col}"::double precision AS value FROM {table_name} WHERE "{col}" IS NOT NULL AND "{col}" > 0')
            union_query = ' UNION ALL '.join(union_parts)
            
            base_query = f"""
            WITH ValidMetabolites AS (
                SELECT name AS metabolite
                FROM {table_name}
                WHERE name IS NOT NULL
                AND {len(valid_bacteria)} = (
                    SELECT COUNT(*)
                    FROM (VALUES {values_clause}) AS t(col_name, value)
                    WHERE value IS NOT NULL AND value > 0
                )
            ),
            SelectedBacteriaUnpivot AS (
                SELECT 
                    vm.metabolite,
                    t.col_name::text AS bacteria,
                    t.value::double precision
                FROM ValidMetabolites vm
                JOIN {table_name} t_data ON vm.metabolite = t_data.name
                CROSS JOIN LATERAL (
                    VALUES {cross_join_values}
                ) AS t(col_name, value)
                WHERE t.value IS NOT NULL AND t.value > 0
            ),
            AllBacteriaRanking AS (
                SELECT 
                    name AS metabolite,
                    bacteria::text,
                    value::double precision,
                    ROW_NUMBER() OVER (PARTITION BY name ORDER BY value DESC) AS rank
                FROM (
                    {union_query}
                ) all_data
            ),
            Top10PerMetabolite AS (
                SELECT metabolite, bacteria, value, rank
                FROM AllBacteriaRanking 
                WHERE rank <= 10
            ),
            MetabolitesWithAllSelected AS (
                SELECT metabolite
                FROM Top10PerMetabolite
                WHERE bacteria = ANY(%s)
                GROUP BY metabolite
                HAVING COUNT(DISTINCT bacteria) = {len(valid_bacteria)}
            )
            SELECT s.metabolite, s.bacteria, s.value, t.rank
            FROM SelectedBacteriaUnpivot s
            JOIN MetabolitesWithAllSelected m USING (metabolite)
            JOIN Top10PerMetabolite t ON s.metabolite = t.metabolite AND s.bacteria::text = t.bacteria::text
            ORDER BY s.metabolite, t.rank;
            """
            
            logging.info("Executing highly optimized cumulative query")
            cursor.execute(base_query, (valid_bacteria,))
            data = cursor.fetchall()
            
            if not data:
                logging.warning("No metabolites found where all selected bacteria appear together in top 10")
                return None

            # Create DataFrame with optimized memory usage
            columns = ['metabolite', 'bacteria', 'value', 'rank']
            result_df = pd.DataFrame(data, columns=columns)
            
            # Convert value to numeric efficiently
            result_df['value'] = pd.to_numeric(result_df['value'], errors='coerce')
            result_df = result_df.dropna(subset=['value'])

            logging.info("Cumulative analysis completed. Shape: %s", result_df.shape)
            return result_df

    except Exception as e:
        logging.error("Error fetching cumulative data: %s", e)
        return None


@memory_logger("get_metabolite_data: Memory")
@log_time("get_metabolite_data: DB Query")
@simple_cache(max_size=100, ttl=300)  # Cache metabolite data for 5 minutes
def get_metabolite_data(table_name, metabolite):
    """Fetch all bacteria values for a specific metabolite"""
    try:
        with get_db_connection() as cursor:
            # Get bacterial columns (only numeric columns, excluding metadata columns)
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND data_type IN ('double precision', 'numeric', 'integer', 'real', 'float', 'decimal')
                AND column_name NOT IN ('name', 'mz', 'rt', 'list_2_match', 'Type', 'Metabolite')
            """, (table_name,))
            bacterial_columns = [row[0] for row in cursor.fetchall()]

            # Create UNPIVOT query with properly quoted column names, casting values to a numeric type
            unpivot_parts = []
            for col in bacterial_columns:
                query_part = f"SELECT name as metabolite, '{col}' as bacteria, " + f'"{col}"' + f"::double precision as value FROM {table_name}"
                unpivot_parts.append(query_part)
            unpivot_query = " UNION ALL ".join(unpivot_parts)
            
            query = f"""
            WITH unpivoted AS ({unpivot_query})
            SELECT metabolite, bacteria, value
            FROM unpivoted 
            WHERE metabolite = %s AND value IS NOT NULL
            """
            
            cursor.execute(query, (metabolite,))
            data = cursor.fetchall()
            
            return pd.DataFrame(data, columns=['metabolite', 'bacteria', 'value'])

    except Exception as e:
        logging.error("Error fetching metabolite data: %s", e)
        return None

@memory_logger("get_bacteria_data: Memory")
@log_time("get_bacteria_data: DB Query")
@simple_cache(max_size=100, ttl=300)  # Cache bacteria data for 5 minutes
def get_bacteria_data(table_name, bacteria):
    """Fetch all metabolite values for a specific bacteria"""
    try:
        with get_db_connection() as cursor:
            # Quote the bacteria column name properly
            quoted_bacteria = f'"{bacteria}"'
            query = f"""
            SELECT name, {quoted_bacteria} as value 
            FROM {table_name} 
            WHERE {quoted_bacteria} IS NOT NULL
            """
            
            cursor.execute(query)
            data = cursor.fetchall()
            
            df = pd.DataFrame(data, columns=['metabolite', 'value'])
            df['bacteria'] = bacteria
            return df

    except Exception as e:
        logging.error("Error fetching bacteria data: %s", e)
        return None                       
            
@log_time("get_column_names: DB Query")
@simple_cache(max_size=20, ttl=600)  # Cache column names for 10 minutes
def get_column_names(table_name):
    """
    Fetches all column names except the 'name' column from the table.

    Args:
        table_name (str): Name of the table.

    Returns:
        list: List of column names.
    """
    try:
        # Columns to exclude
        columns_to_exclude = ['mz', 'rt', 'list_2_match', 'name', 'Type', 'metabolite']
        
        with get_db_connection() as cursor:
            # Check if table exists first
            check_table_query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
            """
            cursor.execute(check_table_query, (table_name,))
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logging.error(f"Table '{table_name}' does not exist in the database.")
                return []
            
            # Dynamically generating the query excluding unwanted columns
            query = f"SELECT * FROM {table_name} LIMIT 0"
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description if desc[0] not in columns_to_exclude]
            
            logging.info(f"Retrieved {len(columns)} column names from {table_name}")
            return columns
        
    except Exception as e:
        logging.error(f"Error retrieving column names from {table_name}: {e}")
        return []


def get_bacteria_names(table_name):
    """
    Fetches all distinct bacteria names (or rows in the 'name' column).

    Args:
        table_name (str): Name of the table.

    Returns:
        list: List of unique bacteria names.
    """
    try:
        with get_db_connection() as cursor:
            # Columns to exclude
            columns_to_exclude = ['mz', 'rt', 'list_2_match']
            
            # Fetch all columns in the table
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'")
            all_columns = [row[0] for row in cursor.fetchall()]

            # Exclude the specified columns
            columns_to_include = [col for col in all_columns if col not in columns_to_exclude]
            
            # Create the query to select distinct names, excluding the unwanted columns
            query = f"SELECT DISTINCT name FROM {table_name} WHERE name IS NOT NULL"
            
            # Execute the query to get distinct bacteria names
            cursor.execute(query)
            names = [row[0] for row in cursor.fetchall()]
            
            return names
    except Exception as e:
        logging.error(f"Error getting bacteria names from table '{table_name}': {e}")
        return []


@memory_logger("get_top_bottom_bacteria_values: Memory")
@log_time("get_top_bottom_bacteria_values: DB Query")
@simple_cache(max_size=100, ttl=300)  # Cache for 5 minutes
def get_top_bottom_bacteria_values(table_name, selected_compound, top_n=10, order="desc"):
    """
    Fetches top/bottom N bacteria values for a selected compound, with processing offloaded to the database.
    
    Args:
        table_name (str): Name of the table in the database.
        selected_compound (str): Name of the compound to filter by.
        top_n (int): Number of values to fetch (default is 10).
        order (str): Fetch order - 'desc' for top values, 'asc' for bottom values.

    Returns:
        pd.DataFrame: Clean DataFrame with top/bottom N bacteria values
    """
    try:
        with get_db_connection() as cursor:
            # Get only numeric bacterial columns
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND data_type IN ('double precision', 'numeric', 'integer', 'real', 'float', 'decimal')
                AND column_name NOT IN ('name', 'mz', 'rt', 'list_2_match', 'Type', 'Metabolite')
            """, (table_name,))
            bacterial_columns = [row[0] for row in cursor.fetchall()]

            if not bacterial_columns:
                logging.warning(f"No bacterial columns found in table {table_name}")
                return pd.DataFrame()

            # Build the unpivot part of the query
            unpivot_parts = []
            for col in bacterial_columns:
                # Ensure column names are quoted to handle special characters
                unpivot_parts.append(f"SELECT name as metabolite, '{col}' as bacteria, \"{col}\"::float as value FROM {table_name}")
            
            unpivot_query = " UNION ALL ".join(unpivot_parts)

            # Build the final query with ranking
            sql_order = "DESC" if order.lower() == "desc" else "ASC"
            
            query = f"""
            WITH UnpivotedData AS (
                {unpivot_query}
            ),
            RankedBacteria AS (
                SELECT
                    metabolite,
                    bacteria,
                    value,
                    ROW_NUMBER() OVER(PARTITION BY metabolite ORDER BY value {sql_order} NULLS LAST) as rn
                FROM UnpivotedData
                WHERE metabolite = %s AND value IS NOT NULL AND value > 0
            )
            SELECT metabolite, bacteria, value
            FROM RankedBacteria
            WHERE rn <= %s
            """
            
            cursor.execute(query, (selected_compound, top_n))
            data = cursor.fetchall()

            if not data:
                logging.warning(f"No data found for compound: {selected_compound}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=['metabolite', 'bacteria', 'value'])
            logging.info(f"Fetched {len(df)} {order.upper()} {top_n} values for {selected_compound} from database.")
            return df

    except Exception as e:
        logging.error(f"Error fetching top/bottom values for {selected_compound}: {e}")
        return pd.DataFrame()

# ===== ENHANCED DATA PROCESSING UTILITIES =====
@memory_logger("clean_dataframe_values: Memory")
@log_time("clean_dataframe_values: Processing")
def clean_dataframe_values(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Vectorized data cleaning for better performance
    
    Args:
        df: DataFrame to clean
        value_col: Column name containing values to clean
        
    Returns:
        Cleaned DataFrame with valid numeric values
    """
    # Remove null values
    df_clean = df[df[value_col].notnull()].copy()
    
    # Convert to numeric, coercing errors to NaN
    df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
    
    # Remove NaN, inf, and -inf values
    df_clean = df_clean[
        df_clean[value_col].notnull() & 
        np.isfinite(df_clean[value_col])
    ]
    
    # Remove zero values if needed (optional)
    df_clean = df_clean[df_clean[value_col] > 0]
    
    return df_clean

@memory_logger("optimize_dataframe_dtypes: Memory")
@log_time("optimize_dataframe_dtypes: Processing")
def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame data types for better memory usage and performance
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        DataFrame with optimized data types
    """
    df_optimized = df.copy()
    
    # Optimize numeric columns
    for col in df_optimized.select_dtypes(include=['float64']):
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    for col in df_optimized.select_dtypes(include=['int64']):
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
    
    # Optimize string columns to category if they have repeated values
    for col in df_optimized.select_dtypes(include=['object']):
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # Less than 50% unique
            df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized

@memory_logger("batch_process_data: Memory")
@log_time("batch_process_data: Processing")
def batch_process_data(data_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Efficiently batch process multiple DataFrames
    
    Args:
        data_list: List of DataFrames to process
        
    Returns:
        Combined and processed DataFrame
    """
    if not data_list:
        return pd.DataFrame()
    
    # Use concat for better performance than iterative append
    combined_df = pd.concat(data_list, ignore_index=True)
    
    # Remove duplicates efficiently
    combined_df = combined_df.drop_duplicates()
    
    # Optimize data types
    combined_df = optimize_dataframe_dtypes(combined_df)
    
    return combined_df









@log_time("get_mz_values: DB Query")
@simple_cache(max_size=50, ttl=900)
def get_mz_values(table_name):
    try:
        with get_db_connection() as cursor:
            query_mz_values = f"SELECT DISTINCT mz FROM {table_name}"
            cursor.execute(query_mz_values)
            mz_values = [row[0] for row in cursor.fetchall()]

            # print("mzval", mz_values[1])
            mz_values = sorted(mz_values, key=lambda s: str(s).casefold() if isinstance(s, str) else s)
            # print("mz_values", mz_values)
            return mz_values
    except Exception as e:
        logging.error(f"Error getting mz values from table '{table_name}': {e}")
        return []


@log_time("get_cecum_and_ascending_mz_values: DB Query")
@simple_cache(max_size=20, ttl=900)
def get_cecum_and_ascending_mz_values(regions):
    try:
        with get_db_connection() as cursor:
            # Initialize an empty set to store the Mz values
            mz_values_set = set()

            # Loop through each region and dynamically generate the SQL query
            for region in regions:
                # SQL query to get Mz values with q_fdr < 0.05 in the specified region
                query = f"SELECT DISTINCT mz FROM {region} WHERE q_fdr <= 0.05"
                cursor.execute(query)
                region_mz_values = {row[0] for row in cursor.fetchall()}

                # If it's the first region, set the Mz values directly
                if not mz_values_set:
                    mz_values_set = region_mz_values
                else:
                    # If it's not the first region, take the intersection with the existing Mz values
                    mz_values_set &= region_mz_values

            mz_values_set = sorted(mz_values_set, key=lambda s: s.casefold())

            return mz_values_set
    except Exception as e:
        logging.error(f"Error getting cecum and ascending mz values: {e}")
        return []


@simple_cache(max_size=20, ttl=900)
def get_one_qfdr_value(all_regions):
    # Get Mz values for each region
    region_mz_values = {region: set(
        get_mz_values(region)) for region in all_regions}

    # Find Mz values with q < 0.05 only in one region (not in other 6)
    unique_specific_subsites_mz = set()
    for current_region in all_regions:
        other_regions = set(all_regions) - {current_region}
        current_region_mz = region_mz_values[current_region]

        # Find Mz values with q < 0.05 in the current region
        current_region_q05_mz = set(get_q05_mz_values(current_region))

        # Find Mz values with q < 0.05 in all other regions
        other_regions_q05_mz = set()
        for other_region in other_regions:
            other_regions_q05_mz |= set(
                get_q05_mz_values(other_region))

        # Find Mz values with q < 0.05 only in the current region (not in other 6)
        specific_subsites_mz = current_region_q05_mz - other_regions_q05_mz

        # Update the set of unique Mz values
        unique_specific_subsites_mz |= specific_subsites_mz

    # Create options and default value
    options = [{"label": mz, "value": mz}
               for mz in sorted(unique_specific_subsites_mz, key=lambda s: s.casefold())]
    # options = sorted(options)

    default_value = sorted(list(unique_specific_subsites_mz))[
        0] if unique_specific_subsites_mz else None

    return options, default_value


@simple_cache(max_size=50, ttl=900)
def get_q05_mz_values(region):
    try:
        with get_db_connection() as cursor:
            query = f"SELECT DISTINCT mz FROM {region} WHERE q_fdr <= 0.05"
            cursor.execute(query)
            q05_mz_values = {row[0] for row in cursor.fetchall()}

            q05_mz_values = sorted(q05_mz_values,key=lambda s: str(s).casefold() if isinstance(s, str) else s)
            
            return q05_mz_values
    except Exception as e:
        logging.error(f"Error getting q05 mz values for region {region}: {e}")
        return []


@simple_cache(max_size=2, ttl=1800)
def get_q05_mz_forest_values():
    try:
        # Establish a connection to the database
        with get_db_connection() as cursor:
            # List of columns to be selected based on the condition
            columns = ["mz"]

            # List of regions
            regions = ["cecum", "ascending", "transverse",
                    "descending", "sigmoid", "rectosigmoid", "rectum"]
            values = []

            # Construct the query for each region separately
            for reg in regions:
                # Construct the column name for the current region's Pvalue column
                pvalue_column = f"Pvalue_{reg}"

                # Construct the query to select distinct mz values where q_fdr <= 0.05 for the current region
                query = f"SELECT DISTINCT mz FROM forest_plot WHERE {pvalue_column} <= 0.05"
                # Add conditions for other regions
                for other_reg in regions:
                    if other_reg != reg:
                        other_pvalue_column = f"Pvalue_{other_reg}"
                        query += f" AND {other_pvalue_column} > 0.05"

                # Execute the query
                cursor.execute(query)
                columns.append(pvalue_column)
                # Fetch all the rows and extract the mz values
                q05_mz_values = {row[0] for row in cursor.fetchall()}
                # print("q05_mz_values", list(q05_mz_values))
                # print("\n")
                values.extend(list(q05_mz_values))
                # print("values", values)

            values = sorted(values, key=lambda s: str(s).casefold() if isinstance(s, str) else s)
            return values
    except Exception as e:
        logging.error(f"Error getting q05 mz forest values: {e}")
        return []


@simple_cache(max_size=20, ttl=900)
def get_linear_values(regions):
    try:
        with get_db_connection() as cursor:
            # Initialize an empty set to store the Mz values
            mz_values_set = set()

            # Loop through each region and dynamically generate the SQL query
            for region in regions:
                query = f"SELECT DISTINCT mz FROM {region} WHERE q_fdr <= 0.05"
                cursor.execute(query)
                region_mz_values = {row[0] for row in cursor.fetchall()}

                # If it's the first region, set the Mz values directly
                if not mz_values_set or region_mz_values:
                    mz_values_set |= region_mz_values

            mz_values_set = sorted(mz_values_set,key=lambda s: str(s).casefold() if isinstance(s, str) else s)

            return mz_values_set
    except Exception as e:
        logging.error(f"Error getting linear values: {e}")
        return []


def get_case_columns_query(table_name, selected_mz):
    try:
        # Connect to the database
        with get_db_connection() as cursor:
            # # Get all column names from the table
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
            all_columns = [desc[0] for desc in cursor.description]
            # print(all_columns)
            # Construct the SQL query dynamically with quoted column names
            case_columns = [f'"{col}"' for col in all_columns if '_case' in col.lower()]
            control_columns = [f'"{col}"' for col in all_columns if '_control' in col.lower()]
            query_case = f"SELECT {', '.join(case_columns)} FROM {table_name} WHERE mz = '{selected_mz}'"
            query_control = f"SELECT {', '.join(control_columns)} FROM {table_name} WHERE mz = '{selected_mz}'"
            get_side_val = f"SELECT q_fdr, log_fc_matched FROM {table_name} WHERE mz = '{selected_mz}'"
            # print("query_case" ,query_case)
            # print("query_control", query_control)

            cursor.execute(query_case)
            case_results = cursor.fetchall()
            # print("heelooo5",case_results)

            cursor.execute(query_control)
            control_results = cursor.fetchall()
            # print("heelooo4",control_results)

            cursor.execute(get_side_val)
            final_get_side_val = cursor.fetchall()
            # print("heelooo6",final_get_side_val)

            # print("heelooo7",case_results, control_results, final_get_side_val)
            return case_results, control_results, final_get_side_val
    except Exception as e:
        logging.error(f"Error in get_case_columns_query for table {table_name} and mz {selected_mz}: {e}")
        return [], [], []


def get_case_columns_vs_query(columName, selected_mz, table_name):
    try:
        # Connect to the database
        with get_db_connection() as cursor:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
            all_columns = [desc[0] for desc in cursor.description]
            # print("all_columns", all_columns)
            case_columns = [f'"{col}"' for col in all_columns if f'case_{columName}_' in col.lower() and 'vs' not in col.lower()]
            query_case = f"SELECT {', '.join(case_columns)} FROM {table_name} WHERE mz = '{selected_mz}'"

            cursor.execute(query_case)
            case_results = cursor.fetchall()

            case_values = [item for sublist in case_results for item in sublist]
            return case_results
    except Exception as e:
        logging.error(f"Error in get_case_columns_vs_query for table {table_name} and mz {selected_mz}: {e}")
        return []


def get_case_columns_linear_query(columName, selected_mz, table_name):
    try:
        # Connect to the database
        with get_db_connection() as cursor:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
            all_columns = [desc[0] for desc in cursor.description]
            # print("all_columns", all_columns)

            case_columns = [f'"{col}"' for col in all_columns if f'case_{columName}_' in col.lower() and 'vs' not in col.lower()]
            query_case = f"SELECT {', '.join(case_columns)} FROM {table_name} WHERE mz = '{selected_mz}'"
            cursor.execute(query_case)
            case_results = cursor.fetchall()

            get_side_val = f"SELECT q_fdr FROM {table_name} WHERE mz = '{selected_mz}'"
            cursor.execute(get_side_val)
            qfdr_results = cursor.fetchall()

            case_values = [item for sublist in case_results for item in sublist]
            return case_results, qfdr_results
    except Exception as e:
        logging.error(f"Error in get_case_columns_linear_query for table {table_name} and mz {selected_mz}: {e}")
        return [], []


def vs_columnNames(table_name, fig, selected_mz, region_call):
    try:
        with get_db_connection() as cursor:
            col_vs = []

            cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
            all_columns = [desc[0] for desc in cursor.description]
            vs_columns = [f'"{col}"' for col in all_columns if 'vs' in col.lower()]
            query_q_vs = f"SELECT {', '.join(vs_columns)} FROM {table_name} WHERE mz = '{selected_mz}'"
            cursor.execute(query_q_vs)
            query_q_vs_result = cursor.fetchall()
            # print("query_q_vs_result", query_q_vs_result)

            for col in all_columns:
                if 'vs' in col.lower():
                    col_vs.append(col)
            # print(col_vs)
            index = 0
            vpos = 0.69
            hpos = 0.7
            for i in range(len(region_call)):
                for j in range(i+1, len(region_call)):
                    vs_value_name = region_call[i]+"_vs_"+region_call[j]
                    vs_value_name_neg = region_call[j]+"_vs_"+region_call[i]
                    # print("vpos", vs_value_name, vs_value_name_neg)

                    if vs_value_name in col_vs:
                        vs_value = col_vs.index(vs_value_name)
                        # print(query_q_vs_result[0][vs_value])
                        qFdr = query_q_vs_result[0][vs_value]
                        # print("exist_", i)

                        if qFdr <= 0.001:
                            qFdrStars = '***'
                            add_comparison_lines(fig, region_call, [region_call[i], region_call[j]], [
                                                vpos+index, hpos+index], symbol=qFdrStars, )
                            index += 0.03
                            # print("vpos", vpos+index, hpos+index)
                        elif qFdr <= 0.01 and qFdr > 0.001:
                            qFdrStars = '**'
                            add_comparison_lines(fig, region_call, [region_call[i], region_call[j]], [
                                                vpos+index, hpos+index], symbol=qFdrStars, )
                            index += 0.03
                            #
                            # ("vpos", vpos+index, hpos+index)

                        elif qFdr <= 0.05 and qFdr > 0.01:
                            qFdrStars = '*'
                            add_comparison_lines(fig, region_call, [region_call[i], region_call[j]], [
                                                vpos+index, hpos+index], symbol=qFdrStars, )
                            index += 0.03
                            # print("vpos", vpos+index, hpos+index)

                    elif vs_value_name_neg in col_vs:
                        vs_value = col_vs.index(vs_value_name_neg)
                        # print(query_q_vs_result[0][vs_value])
                        # print("exist_", i)
                        qFdr = query_q_vs_result[0][vs_value]
                        if qFdr <= 0.001:
                            qFdrStars = '***'
                            add_comparison_lines(fig, region_call, [region_call[i], region_call[j]], [
                                                vpos+index, hpos+index], symbol=qFdrStars)
                            index += 0.03
                            # print("vpos", vpos+index, hpos+index)
                        elif qFdr <= 0.01 and qFdr > 0.001:
                            qFdrStars = '**'
                            add_comparison_lines(fig, region_call, [region_call[i], region_call[j]], [
                                                vpos+index, hpos+index], symbol=qFdrStars)
                            index += 0.03
                            # print("vpos", vpos+index, hpos+index)

                        elif qFdr <= 0.05 and qFdr > 0.01:
                            qFdrStars = '*'
                            add_comparison_lines(fig, region_call, [region_call[i], region_call[j]], [
                                                vpos+index, hpos+index], symbol=qFdrStars)
                            index += 0.03
                            # print("vpos", vpos+index, hpos+index)
    except Exception as e:
        logging.error(f"Error in vs_columnNames for table {table_name} and mz {selected_mz}: {e}")


def add_comparison_lines(fig, region_call, regions, y_range, symbol):
    # print("com,ing here")
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=regions[0],
        y0=y_range[0],
        x1=regions[0],
        y1=y_range[1],
        line=dict(color="black", width=0.5),
    )
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=regions[0],
        y0=y_range[1],
        x1=regions[1],
        y1=y_range[1],
        line=dict(color="black", width=0.5),
    )
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=regions[1],
        y0=y_range[1],
        x1=regions[1],
        y1=y_range[0],
        line=dict(color="black", width=0.5),
    )

    bar_xcoord_map = {x: idx for idx, x in enumerate(region_call)}
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=(bar_xcoord_map[regions[0]] + bar_xcoord_map[regions[1]]) / 2,
            y=y_range[1] * 1.04,
            showarrow=False,
            text=symbol,
            textangle=0,
            xref="x",
            yref="paper",
        )
    )


def get_dropdown_options():
    # image_urls = [
    #     "assets/images/car.jpg",
    #     "assets/images/car1.jpg",
    #     "assets/images/car.jpg"
    # ]
    dropdown_options = [{"label": f"Image {i+1}", "value": image_urls[i]}
                        for i in range(len(image_urls))]
    return dropdown_options


# def forest_plot(selected_mz, regions):
#     connection = psycopg2.connect(db_url)
#     cursor = connection.cursor()
#     table_name = "forest_plot"

#     # Create a list to store dictionaries for all regions
#     result_list = []
#     # regions = ['cecum', 'ascending', 'transverse',
#     #            'descending', 'sigmoid', 'Rectosigmoid', 'Rectum']

#     # Define custom colors foreach region
#     custom_colors = ['red', 'blue', 'green',
#                      'purple', 'orange', 'pink', 'brown']

#     # Iterate over regions
#     for region in regions:
#         hr_column = f'HR_{region}'
#         pvalue_column = f'Pvalue_{region}'
#         low_column = f'Low_{region}'
#         high_column = f'High_{region}'

#         # Execute SQL queries to fetch data for the current region and selected mz
#         cursor.execute(
#             f"SELECT {hr_column}, {low_column}, {high_column}, {pvalue_column} FROM {table_name} WHERE mz = %s", (selected_mz,))
#         result = cursor.fetchone()

#         if result:
#             # Calculate the HR value and its confidence interval
#             hr_value = result[0]
#             low_value = result[1]
#             high_value = result[2]
#             est_hr = f"{hr_value}({low_value} to {high_value})"

#             # Create a dictionary for the current region
#             result_dict = {
#                 'mz': selected_mz,
#                 'region': region,
#                 'HR': hr_value,
#                 'Low': low_value,
#                 'High': high_value,
#                 'Pvalue': result[3],
#                 'est_hr': est_hr,
#             }

#         # Determine qFdrStars1 based on Pvalue
#         if result[3] <= 0.001:
#             result_dict['Pval'] = '***'
#         elif 0.001 < result[3] <= 0.01:
#             result_dict['Pval'] = '**'
#         elif 0.01 < result[3] <= 0.05:
#             result_dict['Pval'] = '*'
#         else:
#             result_dict['Pval'] = ''

#         # print(result[3])
#         result_list.append(result_dict)

#     # print("result", result_list)
#     # result_list = sorted(result_list)
#     return result_list


# def forest_plot_rcc_lcc(selected_mz, regions):
#     connection = psycopg2.connect(db_url)
#     cursor = connection.cursor()
#     table_name = "forest_rcc_lcc_plot"

#     # Create a list to store dictionaries for all regions
#     result_list = []
#     # regions = ['cecum', 'ascending', 'transverse',
#     #            'descending', 'sigmoid', 'Rectosigmoid', 'Rectum']

#     # Define custom colors foreach region
#     custom_colors = ['red', 'blue', 'green',
#                      'purple', 'orange', 'pink', 'brown']

#     # Iterate over regions
#     for region in regions:
#         hr_column = f'HR_{region}'
#         pvalue_column = f'Pvalue_{region}'
#         low_column = f'Low_{region}'
#         high_column = f'High_{region}'

#         # Execute SQL queries to fetch data for the current region and selected mz
#         cursor.execute(
#             f"SELECT {hr_column}, {low_column}, {high_column}, {pvalue_column} FROM {table_name} WHERE mz = %s", (selected_mz,))
#         result = cursor.fetchone()

#         if result:
#             # Calculate the HR value and its confidence interval
#             hr_value = result[0]
#             low_value = result[1]
#             high_value = result[2]
#             est_hr = f"{hr_value}({low_value} to {high_value})"

#             # Create a dictionary for the current region
#             result_dict = {
#                 'mz': selected_mz,
#                 'region': region,
#                 'HR': hr_value,
#                 'Low': low_value,
#                 'High': high_value,
#                 'Pvalue': result[3],
#                 'est_hr': est_hr,
#             }

#         # Determine qFdrStars1 based on Pvalue
#         if result[3] <= 0.001:
#             result_dict['Pval'] = '***'
#         elif 0.001 < result[3] <= 0.01:
#             result_dict['Pval'] = '**'
#         elif 0.01 < result[3] <= 0.05:
#             result_dict['Pval'] = '*'
#         else:
#             result_dict['Pval'] = ''

#         # print(result[3])
#         result_list.append(result_dict)

#     # print("result", result_list)
#     # result_list = sorted(result_list)
#     return result_list


@log_time("get_gmm_name_by_type: DB Query")
def get_gmm_name_by_type(table_name, type_filter="all"):
    """
    Fetches metabolites filtered by type. Handles different table structures:
    - in_vivo: Uses Type column for filtering, returns name column values
    - gmm_test_1: Uses name column suffix parsing for filtering
    
    Args:
        table_name (str): Name of the table in the database.
        type_filter (str): Filter by type - "by_name", "by_positive", "by_negative", or "all"
        
    Returns:
        list: Sorted list of distinct metabolite values filtered by type.
    """
    try:
        with get_db_connection() as cursor:
            # Check if table exists first
            check_table_query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
            """
            cursor.execute(check_table_query, (table_name,))
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logging.error(f"Table '{table_name}' does not exist")
                return []
            
            # Check if Type column exists and has data
            check_columns_query = f"""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = %s AND column_name = 'Type'
            """
            cursor.execute(check_columns_query, (table_name,))
            has_type_column = len(cursor.fetchall()) > 0
            
            if has_type_column and table_name == "in_vivo":
                # Use Type column for in_vivo table
                logging.info(f"Using Type column for filtering in table '{table_name}'")
                
                if type_filter == "all":
                    query_metabolites = f'SELECT DISTINCT "name" FROM "{table_name}" WHERE "name" IS NOT NULL ORDER BY "name"'
                    cursor.execute(query_metabolites)
                elif type_filter == "by_positive":
                    query_metabolites = f'SELECT DISTINCT "name" FROM "{table_name}" WHERE "Type" = %s AND "name" IS NOT NULL ORDER BY "name"'
                    cursor.execute(query_metabolites, ('by_positive',))
                elif type_filter == "by_negative":
                    query_metabolites = f'SELECT DISTINCT "name" FROM "{table_name}" WHERE "Type" = %s AND "name" IS NOT NULL ORDER BY "name"'
                    cursor.execute(query_metabolites, ('by_negative',))
                elif type_filter == "by_name":
                    query_metabolites = f'SELECT DISTINCT "name" FROM "{table_name}" WHERE "Type" = %s AND "name" IS NOT NULL ORDER BY "name"'
                    cursor.execute(query_metabolites, ('by_name',))
                else:
                    query_metabolites = f'SELECT DISTINCT "name" FROM "{table_name}" WHERE "name" IS NOT NULL ORDER BY "name"'
                    cursor.execute(query_metabolites)
                    
                metabolite_values = [row[0] for row in cursor.fetchall()]
                
            else:
                # Use name column suffix parsing for gmm_test_1 table
                logging.info(f"Using name column suffix parsing for table '{table_name}'")
                
                if type_filter == "all":
                    query_metabolites = f'SELECT DISTINCT "name" FROM "{table_name}" WHERE "name" IS NOT NULL ORDER BY "name"'
                    cursor.execute(query_metabolites)
                elif type_filter == "by_positive":
                    query_metabolites = f'SELECT DISTINCT "name" FROM "{table_name}" WHERE "name" LIKE %s AND "name" IS NOT NULL ORDER BY "name"'
                    cursor.execute(query_metabolites, ('%_POS%',))
                elif type_filter == "by_negative":
                    query_metabolites = f'SELECT DISTINCT "name" FROM "{table_name}" WHERE "name" LIKE %s AND "name" IS NOT NULL ORDER BY "name"'
                    cursor.execute(query_metabolites, ('%_NEG%',))
                elif type_filter == "by_name":
                    # For gmm_test_1, "by_name" means metabolites without _POS or _NEG suffixes
                    # But since all records have _POS, this will return empty for gmm_test_1
                    query_metabolites = f'SELECT DISTINCT "name" FROM "{table_name}" WHERE "name" NOT LIKE %s AND "name" NOT LIKE %s AND "name" IS NOT NULL ORDER BY "name"'
                    cursor.execute(query_metabolites, ('%_POS%', '%_NEG%'))
                else:
                    # Default to all if unknown filter
                    query_metabolites = f'SELECT DISTINCT "name" FROM "{table_name}" WHERE "name" IS NOT NULL ORDER BY "name"'
                    cursor.execute(query_metabolites)
                    
                metabolite_values = [row[0] for row in cursor.fetchall()]
            
            
            logging.info(f"Retrieved {len(metabolite_values)} metabolites for type '{type_filter}' from table '{table_name}'")
            return metabolite_values
        
    except Exception as e:
        logging.error(f"Error getting metabolites by type from table '{table_name}': {e}")
        # Fallback to the original function
        return get_gmm_name(table_name)


def debug_table_structure(table_name):
    """
    Debug function to inspect table structure and sample data.
    
    Args:
        table_name (str): Name of the table to inspect.
        
    Returns:
        dict: Information about table structure and sample data.
    """
    try:
        with get_db_connection() as cursor:
            # Check if table exists
            check_table_query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
            """
            cursor.execute(check_table_query, (table_name,))
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                return {"error": f"Table '{table_name}' does not exist"}
            
            # Get table info
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            # Get column info
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position
            """, (table_name,))
            columns_info = cursor.fetchall()
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            sample_data = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            
            # Get distinct metabolite count
            cursor.execute(f'SELECT COUNT(DISTINCT "name") FROM {table_name}')
            unique_metabolites = cursor.fetchone()[0]
            
            # Get bacteria columns (numeric columns excluding metadata)
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND data_type IN ('double precision', 'numeric', 'integer', 'real', 'float', 'decimal')
                AND column_name NOT IN ('name', 'mz', 'rt', 'list_2_match', 'Type', 'metabolite')
            """, (table_name,))
            bacteria_columns = [row[0] for row in cursor.fetchall()]
            
            return {
                "table_name": table_name,
                "exists": table_exists,
                "total_rows": row_count,
                "unique_metabolites": unique_metabolites,
                "total_columns": len(columns_info),
                "bacteria_columns_count": len(bacteria_columns),
                "bacteria_columns": bacteria_columns[:10],  # First 10 bacteria columns
                "column_info": columns_info,
                "sample_data": {
                    "columns": column_names,
                    "rows": sample_data
                }
            }
        
    except Exception as e:
        return {"error": f"Error inspecting table '{table_name}': {str(e)}"}


@memory_logger("get_heatmap_data: Memory")
@log_time("get_heatmap_data: DB Query")
@simple_cache(max_size=50, ttl=600)
def get_heatmap_data(table_name: str, metabolites: List[str], bacteria: List[str]) -> Optional[pd.DataFrame]:
    """
    Fetches and processes data specifically for heatmap generation, optimized for performance.
    
    Args:
        table_name (str): The name of the database table.
        metabolites (List[str]): A list of metabolite names to include (rows).
        bacteria (List[str]): A list of bacteria column names to include (columns).
        
    Returns:
        pd.DataFrame: A transposed DataFrame ready for plotting, or None if no data.
    """
    if not metabolites or not bacteria:
        logging.warning("Heatmap data fetch requires both metabolites and bacteria.")
        return None

    try:
        with get_db_connection() as cursor:
            # Sanitize column names to prevent SQL injection.
            safe_bacteria_cols = [f'"{col}"' for col in bacteria if col.replace('_', '').isalnum()]
            
            if not safe_bacteria_cols:
                logging.error("No valid bacteria columns provided.")
                return None

            # Create the part of the query for "Net Balance"
            net_balance_sql = " + ".join([f"COALESCE({col}, 0)" for col in safe_bacteria_cols])

            # Construct the main query
            query = f"""
            SELECT 
                name,
                {', '.join(safe_bacteria_cols)},
                ({net_balance_sql}) AS "Net Balance"
            FROM {table_name}
            WHERE name = ANY(%s)
            """
            
            logging.info(f"Executing optimized heatmap query on {table_name}.")
            
            # Use psycopg2's list adaptation for the IN clause
            cursor.execute(query, (metabolites,))
            
            data = cursor.fetchall()
            
            if not data:
                logging.warning(f"No data returned from heatmap query for table {table_name}.")
                return None
                
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(data, columns=columns)
            
            # Set index and transpose for heatmap
            df_processed = df.set_index("name").T
            
            # Ensure "Net Balance" is the first row
            if "Net Balance" in df_processed.index:
                net_balance_row = df_processed.loc[["Net Balance"]]
                other_rows = df_processed.drop("Net Balance")
                df_processed = pd.concat([net_balance_row, other_rows])

            logging.info(f"Successfully processed heatmap data for {table_name}. Shape: {df_processed.shape}")
            return df_processed
            
    except Exception as e:
        logging.error(f"Error in get_heatmap_data for table {table_name}: {e}")
        return None

def clear_function_cache(function_name=None):
    """
    Clear cache for specific function or all functions.
    
    Args:
        function_name (str, optional): Name of function to clear cache for. If None, clears all caches.
    """
    if function_name:
        # Clear specific function cache
        if hasattr(globals()[function_name], 'clear_cache'):
            globals()[function_name].clear_cache()
            logging.info(f"Cache cleared for function: {function_name}")
    else:
        # Clear all function caches
        for name, obj in globals().items():
            if hasattr(obj, 'clear_cache'):
                obj.clear_cache()
        logging.info("All function caches cleared")

def get_cache_info():
    """Get information about all cached functions"""
    cache_info = {}
    for name, obj in globals().items():
        if hasattr(obj, 'cache_info'):
            try:
                cache_info[name] = obj.cache_info()
            except:
                pass
    return cache_info

# Database optimization suggestions:
# 1. Create indexes on frequently queried columns:
#    CREATE INDEX idx_name ON table_name(name);
#    CREATE INDEX idx_bacteria_columns ON table_name USING gin(to_tsvector('english', bacteria_column_name));
# 
# 2. Consider partitioning large tables by metabolite type or value ranges
# 
# 3. Use materialized views for complex aggregations that are queried frequently
# 
# 4. Monitor query performance with EXPLAIN ANALYZE
