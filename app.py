import dash
from dash import dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
from compare_tumor.callback import register_callbacks
from layout import main_layout
from layout404 import main_layout404
import logging
import time
import redis
import os

# Redis Cloud configuration - prioritize REDISCLOUD_URL from Heroku add-on
REDIS_URL = os.getenv("REDISCLOUD_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False, socket_connect_timeout=5, socket_timeout=5)

try:
    redis_client.ping()
    print("Redis connection successful!")
    print(f"Connected to Redis at: {REDIS_URL.split('@')[-1] if '@' in REDIS_URL else 'localhost'}")
except Exception as e:
    print(f"Redis connection failed: {e}")
    print("Falling back to in-memory caching only")

# Initialize simple Redis cache system
from compare_tumor.simple_redis_cache import initialize_simple_redis_cache
try:
    redis_cache_instance = initialize_simple_redis_cache(redis_client)
    print("Simple Redis cache system initialized successfully!")
except Exception as e:
    print(f"Simple Redis cache initialization failed: {e}")

# Import optimization components
from compare_tumor.data_functions import (
    db_pool, 
    get_gmm_name, 
    get_column_names,
    get_gmm_name_by_type
)

region = ["cecum", "ascending", "transverse",
          "descending", "sigmoid", "rectosigmoid", "rectum"]

# Configure performance logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

external_stylesheets = ['assets/stylesheet.css', 'dbc.themes.BOOTSTRAP']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
app.title = 'Gut Microbio Metabolomics'
server = app.server

# ===== PERFORMANCE OPTIMIZATION STARTUP =====
def initialize_performance_optimizations():
    """Initialize database pool and warm up caches on startup"""
    logger.info("Initializing performance optimizations...")
    start_time = time.time()
    
    try:
        # Initialize database connection pool
        logger.info("Initializing database connection pool...")
        db_pool.initialize_pool(minconn=3, maxconn=15)
        
        # Warm up critical caches
        logger.info("Warming up caches...")
        warm_cache()
        
        # Check Redis status
        logger.info("Checking Redis cache status...")
        get_redis_status()
        
        initialization_time = time.time() - start_time
        logger.info(f"Performance optimizations initialized successfully in {initialization_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during performance initialization: {e}")
        # Don't fail the app startup, but log the error
        

def warm_cache():
    """Pre-load frequently accessed data into cache"""
    tables = ['gmm_test_1', 'in_vivo']
    type_filters = ['all', 'by_positive', 'by_negative', 'by_name']
    
    for table in tables:
        try:
            # Cache metabolite names for each type
            for type_filter in type_filters:
                get_gmm_name_by_type(table, type_filter)
            
            # Cache bacteria/column names
            get_column_names(table)
            
            logger.info(f"Cache warmed for table: {table}")
            
        except Exception as e:
            logger.warning(f"Failed to warm cache for table {table}: {e}")

def get_redis_status():
    """Get Redis connection status and cache statistics"""
    try:
        from compare_tumor.simple_redis_cache import get_simple_redis_cache
        cache = get_simple_redis_cache()
        stats = cache.get_stats()
        
        logger.info("Redis Cache Statistics:")
        logger.info(f"  - Redis Available: {stats.get('redis_available', False)}")
        logger.info(f"  - Cache Type: {stats.get('cache_type', 'Unknown')}")
        logger.info(f"  - Memory Cache Size: {stats.get('memory_cache_size', 0)}")
        if 'redis_memory_used' in stats:
            logger.info(f"  - Redis Memory Used: {stats['redis_memory_used']}")
        if 'redis_cache_keys' in stats:
            logger.info(f"  - Redis Cache Keys: {stats['redis_cache_keys']}")
        if 'redis_maxmemory_policy' in stats:
            logger.info(f"  - Redis Eviction Policy: {stats['redis_maxmemory_policy']}")
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get Redis status: {e}")
        return None

# Initialize optimizations on startup
initialize_performance_optimizations()


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])


@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/yale-university':
        return main_layout
    elif pathname == '/':
        # Redirect to '/yale-university' when visiting '/'
        return dcc.Location(pathname='/yale-university', id='redirect')
    else:
        return main_layout404

register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

