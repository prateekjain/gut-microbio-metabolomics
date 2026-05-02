import dash
from dash import dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
from flask import redirect, request
from flask_compress import Compress
from compare_tumor.callback import register_callbacks
from layout import main_layout
from layout404 import main_layout404
import logging
import threading
import time
import os

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

# Gzip Flask responses (incl. Dash's JS bundles, /_dash-layout, callback JSON).
# Heroku's router does NOT auto-compress; without this the 3.6 MB plotly.min.js
# transfers uncompressed and takes 30-40s on slow links.
Compress(server)


@server.before_request
def _root_to_yale():
    if request.path == '/':
        return redirect('/yale-university', code=302)

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

# Run warm-up off the boot path so the dyno binds to the port immediately
threading.Thread(target=initialize_performance_optimizations, daemon=True).start()


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])


@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/yale-university':
        return main_layout
    return main_layout404

register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

