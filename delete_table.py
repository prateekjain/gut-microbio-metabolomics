import os
import logging
import psycopg2
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("delete_table_logs.log"),
        logging.StreamHandler(),
    ],
)

# Load environment variables
load_dotenv()

# Heroku PostgreSQL connection details
db_url = os.getenv('DATABASE_URL')
if not db_url:
    logging.error("DATABASE_URL not found in environment variables.")
    exit()

def delete_table(table_name):
    """
    Delete a table from the PostgreSQL database
    
    Args:
        table_name (str): Name of the table to delete
    """
    try:
        logging.info("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        logging.info("Database connection established successfully.")
        
        # Check if table exists
        check_table_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        );
        """
        cur.execute(check_table_query, (table_name,))
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            logging.warning(f"Table '{table_name}' does not exist in the database.")
            return False
            
        # Delete the table
        drop_table_query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
        logging.info(f"Executing DROP TABLE query: {drop_table_query}")
        cur.execute(drop_table_query)
        
        # Commit the transaction
        conn.commit()
        logging.info(f"Table '{table_name}' deleted successfully.")
        return True
        
    except Exception as e:
        logging.error(f"Error deleting table '{table_name}': {e}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        logging.info("Database connection closed.")

if __name__ == '__main__':
    # Specify the table name you want to delete
    table_to_delete = input("Enter the table name to delete: ").strip()
    
    if table_to_delete:
        confirmation = input(f"Are you sure you want to delete table '{table_to_delete}'? (yes/no): ").strip().lower()
        if confirmation == 'yes':
            success = delete_table(table_to_delete)
            if success:
                print(f"Table '{table_to_delete}' has been deleted successfully.")
            else:
                print(f"Failed to delete table '{table_to_delete}'.")
        else:
            print("Table deletion cancelled.")
    else:
        print("No table name provided.") 