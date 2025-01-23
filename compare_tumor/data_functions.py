# data_functions.py
import os
import psycopg2
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from dotenv import load_dotenv
import pandas as pd
from compare_tumor.constant import *
import logging
import psycopg2
import pandas as pd

# Configure logging
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


def selected_mz_cleaning(selected_mz):
    if "'" in selected_mz:
        selected_mz = selected_mz.replace("'", "''")
        # print("updated mz value", selected_mz)
    return selected_mz

def get_gmm_name(table_name):
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()

    query_gmm_name = f"SELECT DISTINCT name FROM {table_name}"
    cursor.execute(query_gmm_name)
    mz_values = [row[0] for row in cursor.fetchall()]
    # print('mz_values', mz_values)
    cursor.close()
    connection.close()
    # print("mzval", mz_values[1])
    mz_values = sorted(mz_values, key=lambda s: str(s).casefold() if isinstance(s, str) else s)
    # print("mz_values", mz_values)
    return mz_values




def get_all_columns_data(table_name, selected_compound):
    """
    Fetches all rows from a specific table for the selected compound.

    Args:
        table_name (str): Name of the table in the database.
        selected_compound (str): Name of the compound to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame containing all rows for the selected compound.
    """
    logging.info("Fetching data from table: %s", table_name)
    print(f"Fetching data from table: {table_name}")  # Debug log

    try:
        # Connect to the database
        connection = psycopg2.connect(db_url)
        cursor = connection.cursor()
        logging.info("Database connection established")
    except Exception as e:
        logging.error("Error connecting to the database: %s", e)
        print(f"Error connecting to database: {e}")  # Debug log
        return None

    try:
        
        # Fetch column names
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
        columns = [desc[0] for desc in cursor.description]
        logging.info("Columns fetched: %s", columns)
        # print(f"Columns: {columns}")  # Debug log

        # Ensure 'name' column exists
        if 'name' not in columns:
            logging.error("'name' column not found in table: %s", table_name)
            print(f"Error: 'name' column not found in table: {table_name}")  # Debug log
            return None

        # Fetch all rows for the selected compound
        # Use parameterized query to prevent SQL injection
        query = f"SELECT * FROM {table_name} WHERE name = %s"
        
        cursor.execute(query, (selected_compound,))
        data = cursor.fetchall()

        if not data:
            logging.warning("No data found for compound: %s", selected_compound)
            print(f"Warning: No data found for compound: {selected_compound}")  # Debug log
            return None

        logging.info("Data fetched successfully for compound: %s", selected_compound)
        print(f"Data fetched successfully for compound: {selected_compound}. Rows: {data}")  # Debug log

        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        columns_to_exclude = [ 'mz', 'rt', 'list_2_match']
        df = df.drop(columns=columns_to_exclude, errors='ignore')
        logging.info("DataFrame created successfully for compound: %s", selected_compound)
        # Remove duplicate rows
        df = df.drop_duplicates()
        logging.info("Duplicate rows removed, resulting DataFrame shape: %s", df.shape)
        # print(f"DataFrame after removing duplicates:\n{df}")  # Debug log

        return df

    except Exception as e:
        logging.error("Error fetching data: %s", e)
        print(f"Error fetching data: {e}")  # Debug log
        return None

    finally:
        # Ensure cursor and connection are closed
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def get_all_columns_data_all_compounds(table_name):
    """
    Fetches all rows and columns from the given table.

    Args:
        table_name (str): Name of the table in the database.

    Returns:
        pd.DataFrame: DataFrame containing all rows and columns.
    """
    logging.info("Fetching data from table: %s", table_name)
    print(f"Fetching data from table: {table_name}")  # Debug log

    try:
        # Connect to the database
        connection = psycopg2.connect(db_url)
        cursor = connection.cursor()
        logging.info("Database connection established")
    except Exception as e:
        logging.error("Error connecting to the database: %s", e)
        print(f"Error connecting to database: {e}")  # Debug log
        return None

    try:
        # Fetch all data
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        data = cursor.fetchall()

        # Get column names
        columns = [desc[0] for desc in cursor.description]
        logging.info("Columns fetched: %s", columns)
        # print(f"Columns: {columns}")  # Debug log

        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        columns_to_exclude = [ 'mz', 'rt', 'list_2_match']
        df = df.drop(columns=columns_to_exclude, errors='ignore')
        logging.info("DataFrame created successfully for the table: %s", table_name)

        # Drop duplicates if any
        df = df.drop_duplicates()
        print(f"DataFrame after removing duplicates:\n{df}")  # Debug log

        return df

    except Exception as e:
        logging.error("Error fetching data: %s", e)
        print(f"Error fetching data: {e}")  # Debug log
        return None

    finally:
        # Ensure cursor and connection are closed
        if cursor:
            cursor.close()
        if connection:
            connection.close()



def get_multiple_bacteria_top_metabolites(table_name, selected_bacteria):
    """
    Fetches all metabolites for the selected bacteria where the bacteria collectively rank among the top 10 producers.

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
        connection = psycopg2.connect(db_url)
        cursor = connection.cursor()

        # Fetch column names dynamically
        cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table_name,))
        all_columns = [row[0] for row in cursor.fetchall()]
        logging.info("Columns fetched from table: %s", all_columns)

        # Exclude non-bacteria columns
        non_bacteria_columns = ["name", "mz", "rt", "list_2_match"]  # Adjust based on your schema
        bacteria_columns = [col for col in all_columns if col not in non_bacteria_columns]
        logging.info("Bacteria columns determined dynamically: %s", bacteria_columns)

        # Build the dynamic UNPIVOT query
        unpivot_query = " UNION ALL ".join(
            [f"SELECT name AS metabolite, '{col}' AS bacteria, {col} AS value FROM {table_name}" for col in bacteria_columns]
        )
        query = f"""
        WITH UnpivotedData AS (
            {unpivot_query}
        ),
        RankedBacteria AS (
            SELECT
                metabolite,
                bacteria,
                value,
                RANK() OVER (PARTITION BY metabolite ORDER BY value DESC) AS rank
            FROM UnpivotedData
        )
        SELECT * FROM RankedBacteria WHERE rank <= 10;
        """
        logging.info("Executing query:\n%s", query)

        cursor.execute(query, (selected_bacteria,))
        data = cursor.fetchall()
        print('data here',data)

        if not data:
            logging.warning("No data found for selected bacteria in the top 10 metabolites: %s", selected_bacteria)
            return None

        # Create DataFrame
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)
        logging.info("Data fetched successfully for selected bacteria in the top 10 metabolites.")
        return df

    except Exception as e:
        logging.error("Error fetching data: %s", e)
        return None

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

            
            
def get_column_names(table_name):
    """
    Fetches all column names except the 'name' column from the table.

    Args:
        table_name (str): Name of the table.

    Returns:
        list: List of column names.
    """
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()
    
    columns_to_exclude = ['mz', 'rt', 'list_2_match', 'name']

    # Establishing connection
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()
    
    # Dynamically generating the query excluding unwanted columns
    query = f"SELECT * FROM {table_name} LIMIT 0"
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description if desc[0] not in columns_to_exclude]
    
    cursor.close()
    connection.close()
    return columns


def get_bacteria_names(table_name):
    """
    Fetches all distinct bacteria names (or rows in the 'name' column).

    Args:
        table_name (str): Name of the table.

    Returns:
        list: List of unique bacteria names.
    """
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()

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
    cursor.close()
    connection.close()
    
    return names


def get_top_bottom_bacteria_values(table_name, selected_compound, top_n=10, order="desc"):
    """
    Fetches the top or bottom N bacteria values for a specific compound from the database.

    Args:
        table_name (str): Name of the table in the database.
        selected_compound (str): Name of the compound to filter by.
        top_n (int): Number of values to fetch (default is 10).
        order (str): Fetch order - 'desc' for top values, 'asc' for bottom values.

    Returns:
        pd.DataFrame: DataFrame containing the top or bottom N bacteria values for the compound.
    """
    try:
        # Connect to the database
        connection = psycopg2.connect(db_url)
        cursor = connection.cursor()
        logging.info("Database connection established")
    except Exception as e:
        logging.error("Error connecting to the database: %s", e)
        return None

    try:
        # Fetch column names
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
        columns = [desc[0] for desc in cursor.description]

        # Filter bacterial columns (exclude non-bacterial columns like 'name', 'mz', 'rt')
        bacterial_columns = [col for col in columns if col not in ['name', 'mz', 'rt', 'list_2_match']]
        # print("bacterial_columns", bacterial_columns)
        if not bacterial_columns:
            logging.error("No bacterial columns found in the table: %s", table_name)
            return None

        # Query to fetch data for the selected compound
        query = f"SELECT name, {', '.join(bacterial_columns)} FROM {table_name} WHERE name = %s"
        cursor.execute(query, (selected_compound,))
        data = cursor.fetchall()

        # Check if data is available
        if not data:
            logging.warning("No data found for compound: %s", selected_compound)
            return None

        # Create DataFrame from fetched data
        df = pd.DataFrame(data, columns=["name"] + bacterial_columns)

        # Melt DataFrame to transform bacterial columns into rows
        df_melted = df.melt(id_vars=["name"], var_name="bacteria", value_name="value")
        # print('df_melted \n', df_melted)
        # Filter rows with non-null values
        df_filtered = df_melted[df_melted["value"].notnull()]

        # Sort by value and fetch top or bottom N
        ascending = True if order.lower() == "asc" else False
        df_sorted = df_filtered.sort_values(by="value", ascending=ascending)

        # Drop duplicates based on 'bacteria' while keeping the highest/lowest value
        df_sorted = df_sorted.drop_duplicates(subset="bacteria", keep="first").head(top_n)
        
        logging.info(f"Fetched {len(df_sorted)} rows for {order.upper()} {top_n} values of compound: {selected_compound}.")
        # print('df_sorted', df_sorted)
        return df_sorted

    except Exception as e:
        logging.error("Error fetching top/bottom values: %s", e)
        return None

    finally:
        # Ensure cursor and connection are closed
        if cursor:
            cursor.close()
        if connection:
            connection.close()









def get_mz_values(table_name):
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()

    query_mz_values = f"SELECT DISTINCT mz FROM {table_name}"
    cursor.execute(query_mz_values)
    mz_values = [row[0] for row in cursor.fetchall()]

    cursor.close()
    connection.close()
    # print("mzval", mz_values[1])
    mz_values = sorted(mz_values, key=lambda s: str(s).casefold() if isinstance(s, str) else s)
    # print("mz_values", mz_values)
    return mz_values


def get_cecum_and_ascending_mz_values(regions):
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()

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

    connection.close()
    mz_values_set = sorted(mz_values_set, key=lambda s: s.casefold())

    return mz_values_set


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


def get_q05_mz_values(region):
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()

    query = f"SELECT DISTINCT mz FROM {region} WHERE q_fdr <= 0.05"
    cursor.execute(query)
    q05_mz_values = {row[0] for row in cursor.fetchall()}

    connection.close()
    q05_mz_values = sorted(q05_mz_values,key=lambda s: str(s).casefold() if isinstance(s, str) else s)
    
    return q05_mz_values


def get_q05_mz_forest_values():
    # Establish a connection to the database
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()

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

    # Close the database connection
    connection.close()
    values = sorted(values, key=lambda s: str(s).casefold() if isinstance(s, str) else s)
    return values


def get_linear_values(regions):
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()

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

    connection.close()
    mz_values_set = sorted(mz_values_set,key=lambda s: str(s).casefold() if isinstance(s, str) else s)

    return mz_values_set


def get_case_columns_query(table_name, selected_mz):
    # Connect to the database
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()
    # # Get all column names from the table
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
    all_columns = [desc[0] for desc in cursor.description]
    # print(all_columns)
    # Construct the SQL query dynamically
    query_case = f"SELECT {', '.join([col for col in all_columns if '_case' in col.lower()])} FROM {table_name} WHERE mz = '{selected_mz}'"
    query_control = f"SELECT {', '.join([col for col in all_columns if '_control' in col.lower()])} FROM {table_name} WHERE mz = '{selected_mz}'"
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

    # Close the cursor and connection
    cursor.close()
    connection.close()
    # print("heelooo7",case_results, control_results, final_get_side_val)
    return case_results, control_results, final_get_side_val


def get_case_columns_vs_query(columName, selected_mz, table_name):
    # Connect to the database
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()

    cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
    all_columns = [desc[0] for desc in cursor.description]
    # print("all_columns", all_columns)
    query_case = f"SELECT {', '.join([col for col in all_columns if f'case_{columName}_' in col.lower() and 'vs' not in col.lower()])} FROM {table_name} WHERE mz = '{selected_mz}'"

    cursor.execute(query_case)
    case_results = cursor.fetchall()

    case_values = [item for sublist in case_results for item in sublist]
    # Close the cursor and connection
    cursor.close()
    connection.close()

    return case_results


def get_case_columns_linear_query(columName, selected_mz, table_name):
    # Connect to the database
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()

    cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
    all_columns = [desc[0] for desc in cursor.description]
    # print("all_columns", all_columns)

    query_case = f"SELECT {', '.join([col for col in all_columns if f'case_{columName}_' in col.lower() and 'vs' not in col.lower()])} FROM {table_name} WHERE mz = '{selected_mz}'"
    cursor.execute(query_case)
    case_results = cursor.fetchall()

    get_side_val = f"SELECT q_fdr FROM {table_name} WHERE mz = '{selected_mz}'"
    cursor.execute(get_side_val)
    qfdr_results = cursor.fetchall()

    case_values = [item for sublist in case_results for item in sublist]
    # Close the cursor and connection
    cursor.close()
    connection.close()

    return case_results, qfdr_results


def vs_columnNames(table_name, fig, selected_mz, region_call):
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()
    col_vs = []

    cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
    all_columns = [desc[0] for desc in cursor.description]
    query_q_vs = f"SELECT {', '.join([col for col in all_columns if 'vs' in col.lower()])} FROM {table_name} WHERE mz = '{selected_mz}'"
    cursor.execute(query_q_vs, (selected_mz,))
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
    cursor.close()
    connection.close()


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


def forest_plot(selected_mz, regions):
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()
    table_name = "forest_plot"

    # Create a list to store dictionaries for all regions
    result_list = []
    # regions = ['cecum', 'ascending', 'transverse',
    #            'descending', 'sigmoid', 'Rectosigmoid', 'Rectum']

    # Define custom colors foreach region
    custom_colors = ['red', 'blue', 'green',
                     'purple', 'orange', 'pink', 'brown']

    # Iterate over regions
    for region in regions:
        hr_column = f'HR_{region}'
        pvalue_column = f'Pvalue_{region}'
        low_column = f'Low_{region}'
        high_column = f'High_{region}'

        # Execute SQL queries to fetch data for the current region and selected mz
        cursor.execute(
            f"SELECT {hr_column}, {low_column}, {high_column}, {pvalue_column} FROM {table_name} WHERE mz = %s", (selected_mz,))
        result = cursor.fetchone()

        if result:
            # Calculate the HR value and its confidence interval
            hr_value = result[0]
            low_value = result[1]
            high_value = result[2]
            est_hr = f"{hr_value}({low_value} to {high_value})"

            # Create a dictionary for the current region
            result_dict = {
                'mz': selected_mz,
                'region': region,
                'HR': hr_value,
                'Low': low_value,
                'High': high_value,
                'Pvalue': result[3],
                'est_hr': est_hr,
            }

        # Determine qFdrStars1 based on Pvalue
        if result[3] <= 0.001:
            result_dict['Pval'] = '***'
        elif 0.001 < result[3] <= 0.01:
            result_dict['Pval'] = '**'
        elif 0.01 < result[3] <= 0.05:
            result_dict['Pval'] = '*'
        else:
            result_dict['Pval'] = ''

        # print(result[3])
        result_list.append(result_dict)

    # print("result", result_list)
    # result_list = sorted(result_list)
    return result_list


def forest_plot_rcc_lcc(selected_mz, regions):
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()
    table_name = "forest_rcc_lcc_plot"

    # Create a list to store dictionaries for all regions
    result_list = []
    # regions = ['cecum', 'ascending', 'transverse',
    #            'descending', 'sigmoid', 'Rectosigmoid', 'Rectum']

    # Define custom colors foreach region
    custom_colors = ['red', 'blue', 'green',
                     'purple', 'orange', 'pink', 'brown']

    # Iterate over regions
    for region in regions:
        hr_column = f'HR_{region}'
        pvalue_column = f'Pvalue_{region}'
        low_column = f'Low_{region}'
        high_column = f'High_{region}'

        # Execute SQL queries to fetch data for the current region and selected mz
        cursor.execute(
            f"SELECT {hr_column}, {low_column}, {high_column}, {pvalue_column} FROM {table_name} WHERE mz = %s", (selected_mz,))
        result = cursor.fetchone()

        if result:
            # Calculate the HR value and its confidence interval
            hr_value = result[0]
            low_value = result[1]
            high_value = result[2]
            est_hr = f"{hr_value}({low_value} to {high_value})"

            # Create a dictionary for the current region
            result_dict = {
                'mz': selected_mz,
                'region': region,
                'HR': hr_value,
                'Low': low_value,
                'High': high_value,
                'Pvalue': result[3],
                'est_hr': est_hr,
            }

        # Determine qFdrStars1 based on Pvalue
        if result[3] <= 0.001:
            result_dict['Pval'] = '***'
        elif 0.001 < result[3] <= 0.01:
            result_dict['Pval'] = '**'
        elif 0.01 < result[3] <= 0.05:
            result_dict['Pval'] = '*'
        else:
            result_dict['Pval'] = ''

        # print(result[3])
        result_list.append(result_dict)

    # print("result", result_list)
    # result_list = sorted(result_list)
    return result_list
