# callback.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import forestplot as fp
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from compare_tumor.constant import *
from dash.exceptions import PreventUpdate
import json
import plotly.tools as tls
from compare_tumor.data_functions import *
import logging
import time
import functools

from compare_tumor.dynamicPlots import tumor_vs_normal_plot, all_regions_plots, comparable_plots, addAnotations, create_dynamic_scatter_plot
matplotlib.use("Agg")  
region = ["cecum", "ascending", "transverse",
          "descending", "sigmoid", "rectosigmoid", "rectum"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Adjust to DEBUG for more verbose logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("app.log")  # Save logs to a file
    ]
)

# ===== PERFORMANCE OPTIMIZATION DECORATORS =====
def performance_logger(func):
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.info(f"Callback {func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"Callback {func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    return wrapper

def prevent_empty_updates(func):
    """Decorator to prevent updates when inputs are None or empty"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if any required arguments are None or empty
        if args and (args[0] is None or (isinstance(args[0], (list, str)) and len(args[0]) == 0)):
            raise PreventUpdate
        return func(*args, **kwargs)
    return wrapper

def register_callbacks(app):
    
    def get_table_name_from_tab(tab_id):
        """
        Helper function to get table name based on tab ID
        Returns the appropriate table name for each tab
        """
        table_mapping = {
            "tab-a": "gmm_test_1",  # In Vitro tab
            "tab-b": "in_vivo",     # In Vivo tab
        }
        return table_mapping.get(tab_id, "gmm_test_1")  # Default fallback

    def get_table_name_for_component(component_id):
        """
        Helper function to determine table name based on component ID suffix
        """
        if component_id.endswith("-a"):
            return "gmm_test_1"  # In Vitro
        elif component_id.endswith("-b"):
            return "in_vivo"     # In Vivo
        else:
            return "gmm_test_1"  # Default fallback
    
    # Callbacks to show/hide details
    @app.callback(
        [
            Output("cohort-details", "style"),
            Output("preparation-details", "style"),
            Output("feature-details", "style"),
            Output("link-details", "style"),
            Output("project-details", "style"),
            Output("contact-details", "style")
        ],
        [
            Input("cohort-option", "n_clicks"),
            Input("preparation-option", "n_clicks"),
            Input("feature-option", "n_clicks"),
            Input("link-option", "n_clicks"),
            Input("project-option", "n_clicks"),
            Input("contact-option", "n_clicks")
            
        ]
    )
    def show_details(n_cohort, n_preparation, n_feature, n_link, n_project, n_contact):
        ctx = dash.callback_context
        if not ctx.triggered:
            return [{"display": "none"}] * 6
        else:
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if triggered_id == "cohort-option":
                return [{"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"},]
            elif triggered_id == "preparation-option":
                return [{"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}]
            elif triggered_id == "feature-option":
                return [{"display": "none"}, {"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}]
            elif triggered_id == "link-option":
                return [{"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"},]
            elif triggered_id == "project-option":
                return [{"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "block"}, {"display": "none"}]
            elif triggered_id == "contact-option":
                return [{"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"},{"display": "none"}, {"display": "block"}]


    @app.callback(
        [Output(f'scatter-plot-mz_minus_h-{i}', 'figure') for i in range(7)],
        [Input('compound-dropdown', 'value')]
    )
    def tumor_vs_normal_m_mins_plots(selected_compound):
        if selected_compound is not None:
            # Fetch and process data based on selected values
            # Assuming you have a column named "mz" in your tables
            selected_mz = float(selected_compound)

            figures = []

            for i in range(len(region)):
                # Fetch data from the database
                query_case, query_control, final_get_side_val = get_case_columns_query(
                    region[i], selected_mz)
                if not query_case or not query_control:
                    figures.append(go.Figure())
                    continue
                query_case = list(query_case[0])
                query_control = list(query_control[0])
                final_get_side_val = list(final_get_side_val[0])

                qFdr = final_get_side_val[0]
                scatter_plot = tumor_vs_normal_plot(
                    query_case, query_control, final_get_side_val,  region[i])

                figures.append(scatter_plot)

            # Show the graph container
            return figures
        else:
            # If dropdown is not selected, hide the container
            return [go.Figure()] * 7


# Callback to update the displayed mz value

    @app.callback(
        [Output("selected-metabolite-gmm-b", "value"),
        Output("selected-bacteria-gmm-b", "value"),
        Output("gmm-scatter-plot-b", "figure")],
        [Input("selected-metabolite-gmm-b", "value"),
        Input("selected-bacteria-gmm-b", "value"),
        Input("top-bottom-radio-b", "value")] 
    )
    def update_scatter_plot_b(selected_metabolite, selected_bacteria, top_bottom):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        logging.info(f"Triggered by (Tab B): {triggered_id}")
        table_name = get_table_name_from_tab("tab-b")

        try:
            # Initialize variables
            df = None
            plot_type = None

            # Handle dropdown triggers
            if triggered_id == "selected-metabolite-gmm-b" and selected_metabolite:
                selected_bacteria = None  # Reset bacteria dropdown
                plot_type = "metabolite"
                
                # Apply radio button filter for metabolite view
                if top_bottom == "top":
                    df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "desc")
                elif top_bottom == "bottom":
                    df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "asc")
                else:  # "all"
                    df = get_metabolite_data(table_name, selected_metabolite)

            elif triggered_id == "selected-bacteria-gmm-b" and selected_bacteria:
                selected_metabolite = None  # Reset metabolite dropdown
                plot_type = "bacteria"
                df = get_bacteria_data(table_name, selected_bacteria)
                
            # Handle radio button trigger
            elif triggered_id == "top-bottom-radio-b":
                logging.info(f"Radio button triggered with metabolite: {selected_metabolite}, filter: {top_bottom}")
                if selected_metabolite:
                    plot_type = "metabolite"
                    if top_bottom == "top":
                        df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "desc")
                        logging.info(f"Top 10 data shape: {df.shape if df is not None else 'None'}")
                    elif top_bottom == "bottom":
                        df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "asc")
                        logging.info(f"Bottom 10 data shape: {df.shape if df is not None else 'None'}")
                    else:  # "all"
                        df = get_metabolite_data(table_name, selected_metabolite)
                        logging.info(f"All data shape: {df.shape if df is not None else 'None'}")
                else:
                    return None, None, create_empty_figure("No Metabolite Selected", 
                                                        "Please select a metabolite first, then choose top/bottom filter")
            else:
                return None, None, create_empty_figure("No Selection", 
                                                    "Please select either a metabolite or bacteria")

            if df is None or df.empty:
                return (selected_metabolite, selected_bacteria, 
                    create_empty_figure("No Data", "No data available for selection"))

            # Create figure based on plot type
            fig = go.Figure()
            
            if plot_type == "metabolite":
                x_axis = df["bacteria"].str.replace("_", " ").str.upper()
                y_axis = df["value"]
                x_title = "Bacteria"
                title = f"Values for Metabolite: {selected_metabolite}"
            else:  # bacteria
                x_axis = df["metabolite"].str.replace("_", " ").str.upper()
                y_axis = df["value"]
                x_title = "Metabolite"
                title = f"Values for Bacteria: {selected_bacteria}"

            scatter_width = max(1000, len(x_axis.unique()) * 20)

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=y_axis,
                    mode="markers",
                    marker=dict(size=max(5, 100 // len(x_axis.unique())), color="#1D78B4"),
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title="Values",
                template="plotly_white",
                width=scatter_width,
                height=400,  # Fixed height for horizontal scrolling
                autosize=False,  # Disable autosize to maintain width for scrolling
                xaxis=dict(
                    tickangle=90,
                    tickfont=dict(size=max(8, min(14, 120 // len(x_axis.unique())))),
                    automargin=True,
                    ticks='outside',
                    ticklen=5,
                    range=[-0.1, len(x_axis.unique())],
                    fixedrange=False  # Allow zooming
                ),
                yaxis=dict(
                    tickfont=dict(color='black'),
                    showline=True,
                    linecolor='black',
                    linewidth=0.1,
                    automargin=True,
                    minor=dict(ticks='outside'),
                    ticks='outside',
                    ticklen=5,
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black',
                    fixedrange=False  # Allow zooming
                ),
                margin=dict(
                    l=60,   # left margin
                    r=40,   # right margin
                    b=max(100, min(200, len(x_axis.unique()) * 2)),  # Dynamic bottom margin
                    t=60,   # top margin
                    pad=4   # padding between axis and labels
                ),
            )

            return selected_metabolite, selected_bacteria, fig

        except Exception as e:
            logging.error("Error in callback: %s", e)
            return None, None, create_empty_figure("Error", str(e))
        
        
    @app.callback(
        [Output("selected-metabolite-gmm-a", "value"),
        Output("selected-bacteria-gmm-a", "value"),
        Output("gmm-scatter-plot-a", "figure")],
        [Input("selected-metabolite-gmm-a", "value"),
        Input("selected-bacteria-gmm-a", "value"),
        Input("top-bottom-radio-a", "value")],
        prevent_initial_call=True  # Prevent initial callback execution
    )
    @performance_logger
    def update_scatter_plot_a(selected_metabolite, selected_bacteria, top_bottom):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        logging.info(f"Triggered by (Tab A): {triggered_id}")
        table_name = get_table_name_from_tab("tab-a")

        try:
            # Initialize variables
            df = None
            plot_type = None

            # Handle dropdown triggers
            if triggered_id == "selected-metabolite-gmm-a" and selected_metabolite:
                selected_bacteria = None  # Reset bacteria dropdown
                plot_type = "metabolite"
                
                # Apply radio button filter for metabolite view
                if top_bottom == "top":
                    df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "desc")
                elif top_bottom == "bottom":
                    df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "asc")
                else:  # "all"
                    df = get_metabolite_data(table_name, selected_metabolite)

            elif triggered_id == "selected-bacteria-gmm-a" and selected_bacteria:
                selected_metabolite = None  # Reset metabolite dropdown
                plot_type = "bacteria"
                df = get_bacteria_data(table_name, selected_bacteria)
                
            # Handle radio button trigger
            elif triggered_id == "top-bottom-radio-a":
                logging.info(f"Radio button triggered with metabolite: {selected_metabolite}, filter: {top_bottom}")
                if selected_metabolite:
                    plot_type = "metabolite"
                    if top_bottom == "top":
                        df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "desc")
                        logging.info(f"Top 10 data shape: {df.shape if df is not None else 'None'}")
                    elif top_bottom == "bottom":
                        df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "asc")
                        logging.info(f"Bottom 10 data shape: {df.shape if df is not None else 'None'}")
                    else:  # "all"
                        df = get_metabolite_data(table_name, selected_metabolite)
                        logging.info(f"All data shape: {df.shape if df is not None else 'None'}")
                else:
                    return None, None, create_empty_figure("No Metabolite Selected", 
                                                        "Please select a metabolite first, then choose top/bottom filter")
            else:
                return None, None, create_empty_figure("No Selection", 
                                                    "Please select either a metabolite or bacteria")

            if df is None or df.empty:
                return (selected_metabolite, selected_bacteria, 
                    create_empty_figure("No Data", "No data available for selection"))

            # Create figure based on plot type
            fig = go.Figure()
            
            if plot_type == "metabolite":
                x_axis = df["bacteria"].str.replace("_", " ").str.upper()
                y_axis = df["value"]
                x_title = "Bacteria"
                title = f"Values for Metabolite: {selected_metabolite}"
            else:  # bacteria
                x_axis = df["metabolite"].str.replace("_", " ").str.upper()
                y_axis = df["value"]
                x_title = "Metabolite"
                title = f"Values for Bacteria: {selected_bacteria}"

            scatter_width = max(1000, len(x_axis.unique()) * 20)

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=y_axis,
                    mode="markers",
                    marker=dict(size=max(5, 100 // len(x_axis.unique())), color="#1D78B4"),
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title="Values",
                template="plotly_white",
                width=scatter_width,
                height=400,  # Fixed height for horizontal scrolling
                autosize=False,  # Disable autosize to maintain width for scrolling
                xaxis=dict(
                    tickangle=90,
                    tickfont=dict(size=max(8, min(14, 120 // len(x_axis.unique())))),
                    automargin=True,
                    ticks='outside',
                    ticklen=5,
                    range=[-0.1, len(x_axis.unique())],
                    fixedrange=False  # Allow zooming
                ),
                yaxis=dict(
                    tickfont=dict(color='black'),
                    showline=True,
                    linecolor='black',
                    linewidth=0.1,
                    automargin=True,
                    minor=dict(ticks='outside'),
                    ticks='outside',
                    ticklen=5,
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black',
                    fixedrange=False  # Allow zooming
                ),
                margin=dict(
                    l=60,   # left margin
                    r=40,   # right margin
                    b=max(100, min(200, len(x_axis.unique()) * 2)),  # Dynamic bottom margin
                    t=60,   # top margin
                    pad=4   # padding between axis and labels
                ),
            )

            return selected_metabolite, selected_bacteria, fig

        except Exception as e:
            logging.error("Error in callback: %s", e)
            return None, None, create_empty_figure("Error", str(e))

        
        
    @app.callback(
        Output('tumor-plot', 'figure'),
        Output('normal-plot', 'figure'),
        [Input('compound-dropdown', 'value')]
    )
    def tumor_normal_m_plus_plot(selected_compound):
        if selected_compound is not None:
            # Fetch and process data based on selected values
            selected_mz = float(selected_compound)
            query_tumor_regions = []
            query_normal_regions = []

            for i in range(len(region)):
                query_case, query_control, final_get_side_val = get_case_columns_query(
                    region[i], selected_mz)

                query_case = list(query_case[0])
                query_control = list(query_control[0])
                query_tumor_regions.extend(query_case)
                query_normal_regions.extend(query_control)

            tumor_plot_all_regions = make_subplots()
            tumor_plot_all_regions = all_regions_plots(
                tumor_plot_all_regions, query_tumor_regions, "Tumor")

            normal_plot_all_regions = make_subplots()
            normal_plot_all_regions = all_regions_plots(
                normal_plot_all_regions, query_normal_regions, "Normal")

            # Show the graph containers
            return tumor_plot_all_regions, normal_plot_all_regions
        else:
            # If dropdown is not selected, hide the containers
            return go.Figure(), go.Figure()

    @app.callback(
        [Output("selected-metabolite-gmm", "value"),
        Output("selected-bacteria-gmm", "value"),
        Output("gmm-scatter-plot", "figure")],
        [Input("selected-metabolite-gmm", "value"),
        Input("selected-bacteria-gmm", "value"),
        Input("top-bottom-radio", "value")] 
    )
    def update_scatter_plot(selected_metabolite, selected_bacteria, top_bottom):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        logging.info(f"Triggered by: {triggered_id}")
        table_name = get_table_name_from_tab("tab-a")

        try:
            # Initialize variables
            df = None
            plot_type = None

            # Handle dropdown triggers
            if triggered_id == "selected-metabolite-gmm" and selected_metabolite:
                selected_bacteria = None  # Reset bacteria dropdown
                plot_type = "metabolite"
                
                # Apply radio button filter for metabolite view
                if top_bottom == "top":
                    df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "desc")
                elif top_bottom == "bottom":
                    df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "asc")
                else:  # "all"
                    df = get_metabolite_data(table_name, selected_metabolite)

            elif triggered_id == "selected-bacteria-gmm" and selected_bacteria:
                selected_metabolite = None  # Reset metabolite dropdown
                plot_type = "bacteria"
                df = get_bacteria_data(table_name, selected_bacteria)
                # Note: Radio buttons don't affect bacteria view as per current logic
                
            # Handle radio button trigger
            elif triggered_id == "top-bottom-radio" and selected_metabolite:
                plot_type = "metabolite"
                if top_bottom == "top":
                    df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "desc")
                elif top_bottom == "bottom":
                    df = get_top_bottom_bacteria_values(table_name, selected_metabolite, 10, "asc")
                else:  # "all"
                    df = get_metabolite_data(table_name, selected_metabolite)
            else:
                return None, None, create_empty_figure("No Selection", 
                                                    "Please select either a metabolite or bacteria")

            if df is None or df.empty:
                return (selected_metabolite, selected_bacteria, 
                    create_empty_figure("No Data", "No data available for selection"))

            # Create figure based on plot type
            fig = go.Figure()
            
            if plot_type == "metabolite":
                x_axis = df["bacteria"].str.replace("_", " ").str.upper()
                y_axis = df["value"]
                x_title = "Bacteria"
                title = f"Values for Metabolite: {selected_metabolite}"
            else:  # bacteria
                x_axis = df["metabolite"].str.replace("_", " ").str.upper()
                y_axis = df["value"]
                x_title = "Metabolite"
                title = f"Values for Bacteria: {selected_bacteria}"

            scatter_width = max(1000, len(x_axis.unique()) * 20)

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=y_axis,
                    mode="markers",
                    marker=dict(size=max(5, 100 // len(x_axis.unique())), color="#1D78B4"),
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title="Values",
                template="plotly_white",
                width=scatter_width,
                height=400,  # Fixed height for horizontal scrolling
                autosize=False,  # Disable autosize to maintain width for scrolling
                xaxis=dict(
                    tickangle=90,
                    tickfont=dict(size=max(8, min(14, 120 // len(x_axis.unique())))),
                    automargin=True,
                    ticks='outside',
                    ticklen=5,
                    range=[-0.1, len(x_axis.unique())],
                    fixedrange=False  # Allow zooming
                ),
                yaxis=dict(
                    tickfont=dict(color='black'),
                    showline=True,
                    linecolor='black',
                    linewidth=0.1,
                    automargin=True,
                    minor=dict(ticks='outside'),
                    ticks='outside',
                    ticklen=5,
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black',
                    fixedrange=False  # Allow zooming
                ),
                margin=dict(
                    l=60,   # left margin
                    r=40,   # right margin
                    b=max(100, min(200, len(x_axis.unique()) * 2)),  # Dynamic bottom margin
                    t=60,   # top margin
                    pad=4   # padding between axis and labels
                ),
            )

            return selected_metabolite, selected_bacteria, fig

        except Exception as e:
            logging.error("Error in callback: %s", e)
            return None, None, create_empty_figure("Error", str(e))



    @app.callback(
        Output("gmm-scatter-top-plot", "figure"),
        [Input("selected-bacteria-top", "value")],
    )
    def update_scatter_top_plot(selected_bacteria):
        logging.info(f"Triggered callback with bacteria: {selected_bacteria}")

        table_name = get_table_name_from_tab("tab-a")

        try:
            if not selected_bacteria:
                logging.info("No bacteria selected.")
                fig = go.Figure()
                fig.update_layout(
                    title="No Bacteria Selected",
                    annotations=[
                        dict(
                            text="Please select bacteria from the dropdown to view the plot.",
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=16),
                            x=0.5,
                            y=0.5,
                        )
                    ],
                    template="plotly_white",
                )
                return fig
            # Fetch data for all bacteria in the top 10 metabolites
            df = get_multiple_bacteria_top_metabolites(table_name, selected_bacteria)
            print('top 10 df',df)
            # Handle edge cases
            if df is None or df.empty:
                logging.warning("No data available for scatter plot.")
                fig = go.Figure()
                fig.update_layout(
                    title="No Data Available",
                    annotations=[
                        dict(
                            text="No data found for the selected bacteria.",
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=16),
                            x=0.5,
                            y=0.5,
                        )
                    ],
                    template="plotly_white",
                )
                return fig

            # Filter the DataFrame to include only the selected bacteria
            if selected_bacteria:
                df = df[df["bacteria"].isin(selected_bacteria)]

            if df.empty:
                logging.info("Selected bacteria do not meet the conditions.")
                fig = go.Figure()
                fig.update_layout(
                    title="No Data Available for Selected Bacteria",
                    annotations=[
                        dict(
                            text="Please select any other bacteria.",
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=16),
                            x=0.5,
                            y=0.5,
                        )
                    ],
                    template="plotly_white",
                )
                return fig

            # Prepare the scatter plot
            fig = go.Figure()

            # Group data by bacteria and metabolite for plotting
            for bacteria, group in df.groupby("bacteria"):
                fig.add_trace(
                    go.Scatter(
                        x=group["metabolite"],
                        y=group["value"],
                        mode="markers",
                        marker=dict(size=10),
                        name=bacteria,
                    )
                )

            # Calculate dynamic width based on the number of metabolites
            num_metabolites = len(df["metabolite"].unique())
            plot_width = max(800, num_metabolites * 40)

            # Update layout
            fig.update_layout(
                title="Scatter Plot of Top 10 Metabolites for Selected Bacteria",
                xaxis_title="Metabolite",
                yaxis_title="Value",
                template="plotly_white",
                width=plot_width,
                xaxis=dict(tickangle=90, showgrid=True),
                yaxis=dict(showgrid=True),
                legend_title="Bacteria",
                showlegend=True,  # Force show legend
                legend=dict(
                    y=1.15,
                    x=1,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1
                ),
                margin=dict(
                    t=100,
                    b=50,
                    l=50,
                    r=50
                )
            )

            return fig

        except Exception as e:
            logging.error("Error in callback: %s", e)
            return go.Figure()

    
    @app.callback(
        Output("gmm-scatter-cumm-top-plot", "figure"),
        [Input("selected-bacteria-cum-top", "value")],
    )
    def update_scatter_top_plot(selected_bacteria):
        logging.info(f"Triggered callback with bacteria: {selected_bacteria}")
        table_name = get_table_name_from_tab("tab-a")

        try:
            # Check for minimum 2 bacteria selection
            if not selected_bacteria or len(selected_bacteria) < 2:
                return create_empty_figure(
                    "Insufficient Selection", 
                    "Please select at least 2 bacteria to compare their collective presence in top producers."
                )

            df = get_multiple_bacteria_cumm_top_metabolites(table_name, selected_bacteria)

            if df is None or df.empty:
                return create_empty_figure(
                    "No Matching Data", 
                    f"The selected bacteria ({', '.join(selected_bacteria)}) are not collectively in the top 10 producers for any metabolite."
                )

            # Create scatter plot with improvements
            fig = go.Figure()

            # Sort metabolites by average value to improve readability
            # Modified to handle duplicates
            metabolite_order = (df.groupby('metabolite')['value']
                            .mean()
                            .sort_values(ascending=False)
                            .index.unique())  # Add unique() to handle duplicates

            bacteria_list = sorted(df['bacteria'].unique())
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            color_map = dict(zip(bacteria_list, colors[:len(bacteria_list)]))

            for bacteria in bacteria_list:
                group = df[df['bacteria'] == bacteria]
                
                # Modified to handle duplicates without reindexing
                group = group.sort_values('metabolite').drop_duplicates(['metabolite', 'bacteria'])
                
                fig.add_trace(
                    go.Scatter(
                        x=group["metabolite"],
                        y=group["value"],
                        mode="markers",
                        marker=dict(
                            size=12,
                            symbol='circle',
                            color=color_map[bacteria],
                            line=dict(width=1, color='white')
                        ),
                        name=bacteria.replace('_', ' ').title(),
                        hovertemplate=(
                            "<b>Bacteria:</b> %{fullData.name}<br>" +
                            "<b>Metabolite:</b> %{x}<br>" +
                            "<b>Value:</b> %{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    )
                )

            scatter_width = max(1000, len(metabolite_order) * 50)
            
            # Rest of the layout remains the same
            fig.update_layout(
                title={
                    'text': "Metabolites where Selected Bacteria are Collectively in Top 10 Producers",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Metabolite",
                yaxis_title="Value",
                template="plotly_white",
                width=scatter_width,
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=1,
                    xanchor="right",
                    x=1.15,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                ),
                xaxis=dict(
                    tickangle=45,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    tickfont=dict(size=10),
                    range=[-0.5, len(metabolite_order) - 0.5]
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black'
                ),
                plot_bgcolor='white'
            )

            return fig

        except Exception as e:
            logging.error("Error in callback: %s", e)
            return create_empty_figure("Error", str(e))

    def create_empty_figure(title, message):
        """Helper function to create empty figure with message"""
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[
                dict(
                    text=message,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                    x=0.5,
                    y=0.5,
                )
            ],
            template="plotly_white",
        )
        return fig
    
    def create_dynamic_scatter_plot(data, plot_type="metabolite", title="", top_bottom=None):
        """
        Enhanced scatter plot creation with dynamic sizing and better performance
        
        Args:
            data: DataFrame with columns ['metabolite', 'bacteria', 'value']
            plot_type: 'metabolite' or 'bacteria' 
            title: Plot title
            top_bottom: Filter for top/bottom values
            
        Returns:
            Plotly figure
        """
        if data is None or data.empty:
            return create_empty_figure("No Data", "No data available for selection")
        
        # Data processing optimizations
        data_clean = data.dropna()
        
        if plot_type == "metabolite":
            x_axis = data_clean["bacteria"].str.replace("_", " ").str.upper()
            y_axis = data_clean["value"]
            x_title = "Bacteria"
            title = title or f"Values for Metabolite"
        else:  # bacteria
            x_axis = data_clean["metabolite"].str.replace("_", " ").str.upper() 
            y_axis = data_clean["value"]
            x_title = "Metabolite"
            title = title or f"Values for Bacteria"

        # Dynamic sizing based on data
        num_points = len(x_axis.unique())
        scatter_width = max(800, min(1400, num_points * 25))
        scatter_height = max(500, min(800, 400 + num_points * 5))
        
        # Create optimized scatter plot
        fig = go.Figure()
        
        # Enhanced markers with better visual encoding
        marker_size = max(6, min(15, 100 // num_points))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] * (num_points // 5 + 1)
        
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_axis,
                mode="markers",
                marker=dict(
                    size=marker_size, 
                    color=colors[:len(x_axis)],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                hovertemplate=(
                    f"<b>{x_title}</b>: %{{x}}<br>" +
                    "<b>Value</b>: %{y:.2f}<br>" +
                    "<extra></extra>"
                )
            )
        )

        # Enhanced layout with better responsiveness
        fig.update_layout(
            title={
                'text': f"<b>{title}</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis=dict(
                title=x_title,
                tickangle=45,
                tickfont=dict(size=max(10, min(14, 80 // num_points))),
                automargin=True,
                ticks='outside',
                ticklen=5,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                range=[-0.5, num_points - 0.5]
            ),
            yaxis=dict(
                title="Values",
                tickfont=dict(color='black'),
                showline=True,
                linecolor='black',
                linewidth=1,
                automargin=True,
                ticks='outside',
                ticklen=5,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            template="plotly_white",
            width=scatter_width,
            height=scatter_height,
            margin=dict(l=80, r=60, b=120, t=80, pad=4),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig
    
    def process_heatmap_data_optimized(df, selected_metabolites=None, selected_bacteria=None):
        """
        Optimized data processing for heatmap generation
        
        Args:
            df: Source DataFrame
            selected_metabolites: List of metabolites to include
            selected_bacteria: List of bacteria to include
            
        Returns:
            Processed DataFrame ready for heatmap
        """
        if df is None or df.empty:
            return None
            
        # Efficient filtering
        df_filtered = df.copy()
        
        if selected_metabolites:
            df_filtered = df_filtered[df_filtered["name"].isin(selected_metabolites)]
        
        if selected_bacteria:
            # Keep 'name' column plus selected bacteria columns
            columns_to_keep = ["name"] + [col for col in selected_bacteria if col in df_filtered.columns]
            df_filtered = df_filtered[columns_to_keep]
        
        if df_filtered.empty:
            return None
            
        # Optimized numeric conversion
        numeric_df = df_filtered.set_index("name")
        numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
        
        # Efficient aggregation 
        sum_row = numeric_df.sum(axis=1, skipna=True)
        numeric_df["Net Balance"] = sum_row
        
        # Transpose for heatmap (metabolites as columns, bacteria as rows)
        heatmap_data = numeric_df.T
        
        return heatmap_data
    
    
    @app.callback(
        Output('gmm-heatmap-plot', 'figure'),
        [
            Input("selected-metabolites", "value"),
            Input("selected-bacteria", "value"),
        ]
    )
    def gmm_heatmap_multiple(selected_bacteria_cols, selected_metabolite_names):
        logging.info(f"Selected Bacteria: {selected_bacteria_cols}, Selected Metabolites: {selected_metabolite_names}")
        
        table_name = get_table_name_from_tab("tab-a")

        if not selected_bacteria_cols or not selected_metabolite_names:
            return create_empty_figure("No Selection", "Please select both bacteria and metabolites.")

        try:
            heatmap_data = get_heatmap_data(table_name, selected_metabolite_names, selected_bacteria_cols)
            if heatmap_data is None or heatmap_data.empty:
                return create_empty_figure("No Data", "No data found for the selected combination.")
        except Exception as e:
            logging.error("Error fetching heatmap data: %s", e)
            return create_empty_figure("Error", "An error occurred while fetching data.")

        # Create heatmap
        try:
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,  # Heatmap values
                x=heatmap_data.columns,  # Metabolites
                y=heatmap_data.index,    # Bacteria, including "Net Balance"
                colorscale='RdYlGn', # Spectral
                colorbar=dict(title="Value"),
                text=heatmap_data.values,  # Add the cell values
                texttemplate="%{text:.2f}",  # Format text to show two decimal places
                textfont={"size": 10},  # Adjust text font size
            ))

            # Calculate dynamic dimensions based on data size
            num_metabolites = len(heatmap_data.columns)
            num_bacteria = len(heatmap_data.index)
            
            # Dynamic width calculation (minimum 800px, 60px per metabolite)
            plot_width = max(800, num_metabolites * 60)
            # Dynamic height calculation (minimum 600px, 40px per bacteria)
            plot_height = max(600, num_bacteria * 40)
            
            print(f"[DEBUG] In Vitro Plot dimensions: {plot_width}x{plot_height} for {num_metabolites} metabolites x {num_bacteria} bacteria")

            # Update layout with responsive sizing
            fig.update_layout(
                title='In Vitro Heatmap: Selected Metabolites and Bacteria',
                xaxis_title='Metabolites',
                yaxis_title='Bacteria',
                width=plot_width,  # Dynamic width
                height=plot_height,  # Dynamic height
                autosize=True,  # Enable responsive sizing
                xaxis=dict(
                    tickangle=90,
                    side='bottom',
                    tickfont=dict(size=max(8, min(12, 100 // num_metabolites))),
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    fixedrange=False  # Allow zooming
                ),
                yaxis=dict(
                    tickfont=dict(color='black', size=max(8, min(12, 100 // num_bacteria))),
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    fixedrange=False  # Allow zooming
                ),
                margin=dict(l=max(100, min(200, num_bacteria * 8)), r=50, t=80, b=max(80, min(150, num_metabolites * 3))),  # Dynamic margins
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=10),
                showlegend=False,  # Heatmap doesn't need legend
            )
            return fig

        except Exception as e:
            logging.error("Error creating heatmap: %s", e)
            return go.Figure()

    # Callback for In Vivo Heatmap (gmm-heatmap-plot-b)
    @app.callback(
        Output('gmm-heatmap-plot-b', 'figure'),
        [
            Input("selected-bacteria-heatmap-b", "value"),
            Input("selected-metabolites-heatmap-b", "value"),
        ]
    )
    def gmm_heatmap_multiple_b(selected_bacteria_cols, selected_metabolite_names):
        logging.info(f"In Vivo Heatmap - Selected Bacteria: {selected_bacteria_cols}, Selected Metabolites: {selected_metabolite_names}")
        
        table_name = "in_vivo"

        if not selected_bacteria_cols or not selected_metabolite_names:
            return create_empty_figure("No Selection", "Please select both bacteria and metabolites.")

        try:
            heatmap_data = get_heatmap_data(table_name, selected_metabolite_names, selected_bacteria_cols)

            if heatmap_data is None or heatmap_data.empty:
                return create_empty_figure("No Data", "No data found for the selected combination.")
        except Exception as e:
            logging.error("Error fetching heatmap data for In Vivo: %s", e)
            return create_empty_figure("Error", "An error occurred while fetching data.")

        # Prepare heatmap data
        try:
            # The data is already processed by get_heatmap_data
            pass
        except Exception as e:
            logging.error("Error preparing data for heatmap: %s", e)
            return create_empty_figure("Error", "Error preparing data for heatmap.")

        # Create heatmap
        try:
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,  # Heatmap values
                x=heatmap_data.columns,  # Metabolites (X-axis)
                y=heatmap_data.index,    # Bacteria (Y-axis), including "Net Balance"
                colorscale='RdYlGn',
                colorbar=dict(title="Value"),
                text=heatmap_data.values,  # Add the cell values
                texttemplate="%{text:.2f}",  # Format text to show two decimal places
                textfont={"size": 10},  # Adjust text font size
            ))

            # Calculate dynamic dimensions based on data size
            num_metabolites = len(heatmap_data.columns)
            num_bacteria = len(heatmap_data.index)
            
            # Dynamic width calculation (minimum 800px, 60px per metabolite)
            plot_width = max(800, num_metabolites * 60)
            # Dynamic height calculation (minimum 600px, 40px per bacteria)
            plot_height = max(600, num_bacteria * 40)
            
            print(f"[DEBUG] Plot dimensions: {plot_width}x{plot_height} for {num_metabolites} metabolites x {num_bacteria} bacteria")

            # Update layout with responsive sizing
            fig.update_layout(
                title='In Vivo Heatmap: Bacteria (Y-axis) vs Metabolites (X-axis)',
                xaxis_title='Metabolites',
                yaxis_title='Bacteria',
                width=plot_width,  # Dynamic width
                height=plot_height,  # Dynamic height
                autosize=True,  # Enable responsive sizing
                xaxis=dict(
                    tickangle=90,
                    side='bottom',
                    tickfont=dict(size=max(8, min(12, 100 // num_metabolites))),
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    fixedrange=False  # Allow zooming
                ),
                yaxis=dict(
                    tickfont=dict(color='black', size=max(8, min(12, 100 // num_bacteria))),
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    fixedrange=False  # Allow zooming
                ),
                margin=dict(l=max(100, min(200, num_bacteria * 8)), r=50, t=80, b=max(80, min(150, num_metabolites * 3))),  # Dynamic margins
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=10),
                showlegend=False,  # Heatmap doesn't need legend
            )
            print(f"[DEBUG] Heatmap created successfully")
            return fig

        except Exception as e:
            logging.error("Error creating heatmap: %s", e)
            print(f"[DEBUG] Error creating heatmap: {e}")
            return create_empty_figure("Error", f"Error creating heatmap: {str(e)}")

    # Callback to update metabolite dropdown options based on type filter for Tab A
    @app.callback(
        Output("selected-metabolite-gmm-a", "options"),
        [Input("type-filter-radio-a", "value")]
    )
    def update_metabolite_options_a(type_filter):
        try:
            from .data_functions import get_gmm_name_by_type
            table_name = get_table_name_for_component("selected-metabolite-gmm-a")
            metabolites = get_gmm_name_by_type(table_name, type_filter)
            return [{"label": name, "value": name} for name in metabolites]
        except Exception as e:
            logging.error(f"Error updating metabolite options for Tab A: {e}")
            return []

    # Callback to update metabolite dropdown options based on type filter for Tab B
    @app.callback(
        Output("selected-metabolite-gmm-b", "options"),
        [Input("type-filter-radio-b", "value")]
    )
    def update_metabolite_options_b(type_filter):
        try:
            from .data_functions import get_gmm_name_by_type
            table_name = get_table_name_for_component("selected-metabolite-gmm-b")
            metabolites = get_gmm_name_by_type(table_name, type_filter)
            return [{"label": name, "value": name} for name in metabolites]
        except Exception as e:
            logging.error(f"Error updating metabolite options for Tab B: {e}")
            return []

    # Callback to update metabolite dropdown options based on type filter for In Vitro Heatmap
    @app.callback(
        Output("selected-bacteria", "options"),
        [Input("type-filter-radio-heatmap", "value")]
    )
    def update_metabolite_options_heatmap_a(type_filter):
        try:
            from .data_functions import get_gmm_name_by_type
            table_name = "gmm_test_1"
            metabolites = get_gmm_name_by_type(table_name, type_filter)
            print(f"[DEBUG] Heatmap A metabolite options updated. Type: {type_filter}, Count: {len(metabolites)}")
            return [{"label": name, "value": name} for name in metabolites]
        except Exception as e:
            logging.error(f"Error updating metabolite options for In Vitro Heatmap: {e}")
            print(f"[DEBUG] Error updating metabolite options for In Vitro Heatmap: {e}")
            return []

    # Callback to update metabolite dropdown options based on type filter for In Vivo Heatmap
    @app.callback(
        Output("selected-metabolites-heatmap-b", "options"),
        [Input("type-filter-radio-heatmap-b", "value")]
    )
    def update_metabolite_options_heatmap_b(type_filter):
        try:
            from .data_functions import get_gmm_name_by_type
            table_name = "in_vivo"
            metabolites = get_gmm_name_by_type(table_name, type_filter)
            print(f"[DEBUG] Heatmap B metabolite options updated. Type: {type_filter}, Count: {len(metabolites)}")
            return [{"label": name, "value": name} for name in metabolites]
        except Exception as e:
            logging.error(f"Error updating metabolite options for In Vivo Heatmap: {e}")
            print(f"[DEBUG] Error updating metabolite options for In Vivo Heatmap: {e}")
            return []

    # Callback for In Vivo Top Metabolites Analysis
    @app.callback(
        Output("gmm-scatter-top-plot-b", "figure"),
        [Input("selected-bacteria-top-b", "value")],
    )
    def update_scatter_top_plot_b(selected_bacteria):
        logging.info(f"Triggered In Vivo top metabolites callback with bacteria: {selected_bacteria}")
        print(f"[DEBUG] In Vivo Top Metabolites - Selected bacteria: {selected_bacteria}")

        table_name = "in_vivo"  # Use in_vivo table

        try:
            if not selected_bacteria:
                logging.info("No bacteria selected for In Vivo top metabolites.")
                return create_empty_figure("No Bacteria Selected", "Please select bacteria from the dropdown to view the plot.")
            
            # Fetch data for all bacteria in the top 10 metabolites
            df = get_multiple_bacteria_top_metabolites(table_name, selected_bacteria)
            print(f'[DEBUG] In Vivo top 10 df: {df.shape if df is not None else "None"}')

            if df is None or df.empty:
                logging.warning("No data found for selected bacteria in In Vivo top metabolites")
                return create_empty_figure("No Data", f"No data found for selected bacteria: {', '.join(selected_bacteria)}")

            # Filter the DataFrame to include only the selected bacteria
            if selected_bacteria:
                df = df[df["bacteria"].isin(selected_bacteria)]
            
            if df.empty:
                return create_empty_figure(
                    "No Data Available for Selected Bacteria",
                    "The selected bacteria are not in the top 10 for any metabolite."
                )

            # Create scatter plot
            fig = go.Figure()

            # Group data by bacteria and metabolite for plotting
            for bacteria, group in df.groupby("bacteria"):
                fig.add_trace(
                    go.Scatter(
                        x=group["metabolite"],
                        y=group["value"],
                        mode="markers",
                        marker=dict(size=10),
                        name=bacteria,
                    )
                )

            # Calculate dynamic width based on the number of metabolites
            num_metabolites = len(df["metabolite"].unique())
            plot_width = max(800, num_metabolites * 40)

            # Update layout
            fig.update_layout(
                title="In Vivo: Top 10 Metabolites for Selected Bacteria",
                xaxis_title="Metabolite",
                yaxis_title="Value",
                template="plotly_white",
                width=plot_width,
                xaxis=dict(tickangle=90, showgrid=True),
                yaxis=dict(showgrid=True),
                legend_title="Bacteria",
                showlegend=True,
                legend=dict(
                    y=1.15,
                    x=1,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1
                ),
                margin=dict(
                    t=100,
                    b=50,
                    l=50,
                    r=50
                )
            )

            return fig

        except Exception as e:
            logging.error("Error in In Vivo top metabolites callback: %s", e)
            print(f"[DEBUG] Error in In Vivo top metabolites callback: {e}")
            return create_empty_figure("Error", str(e))

    # Callback for In Vivo Cumulative Top Metabolites Analysis
    @app.callback(
        Output("gmm-scatter-cumm-top-plot-b", "figure"),
        [Input("selected-bacteria-cum-top-b", "value")],
    )
    def update_scatter_cumm_top_plot_b(selected_bacteria):
        logging.info(f"Triggered In Vivo cumulative top metabolites callback with bacteria: {selected_bacteria}")
        print(f"[DEBUG] In Vivo Cumulative Top Metabolites - Selected bacteria: {selected_bacteria}")
        
        table_name = "in_vivo"  # Use in_vivo table

        try:
            # Check for minimum 2 bacteria selection
            if not selected_bacteria or len(selected_bacteria) < 2:
                return create_empty_figure(
                    "Insufficient Selection", 
                    "Please select at least 2 bacteria to compare their collective presence in top producers."
                )

            df = get_multiple_bacteria_cumm_top_metabolites(table_name, selected_bacteria)

            if df is None or df.empty:
                return create_empty_figure(
                    "No Matching Data", 
                    f"The selected bacteria ({', '.join(selected_bacteria)}) are not collectively in the top 10 producers for any metabolite in In Vivo data."
                )

            # Create scatter plot with improvements
            fig = go.Figure()

            # Sort metabolites by average value to improve readability
            metabolite_order = (df.groupby('metabolite')['value']
                            .mean()
                            .sort_values(ascending=False)
                            .index.unique())

            bacteria_list = sorted(df['bacteria'].unique())
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            color_map = dict(zip(bacteria_list, colors[:len(bacteria_list)]))

            for bacteria in bacteria_list:
                group = df[df['bacteria'] == bacteria]
                
                # Handle duplicates without reindexing
                group = group.sort_values('metabolite').drop_duplicates(['metabolite', 'bacteria'])
                
                fig.add_trace(
                    go.Scatter(
                        x=group["metabolite"],
                        y=group["value"],
                        mode="markers",
                        marker=dict(
                            size=12,
                            symbol='circle',
                            color=color_map[bacteria],
                            line=dict(width=1, color='white')
                        ),
                        name=bacteria.replace('_', ' ').title(),
                        hovertemplate=(
                            "<b>Bacteria:</b> %{fullData.name}<br>" +
                            "<b>Metabolite:</b> %{x}<br>" +
                            "<b>Value:</b> %{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    )
                )

            scatter_width = max(1000, len(metabolite_order) * 50)
            
            fig.update_layout(
                title={
                    'text': "In Vivo: Metabolites where Selected Bacteria are Collectively in Top 10 Producers",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Metabolite",
                yaxis_title="Value",
                template="plotly_white",
                width=scatter_width,
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=1,
                    xanchor="right",
                    x=1.15,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                ),
                xaxis=dict(
                    tickangle=45,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    tickfont=dict(size=10),
                    range=[-0.5, len(metabolite_order) - 0.5]
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black'
                ),
                plot_bgcolor='white'
            )

            return fig

        except Exception as e:
            logging.error("Error in In Vivo cumulative top metabolites callback: %s", e)
            print(f"[DEBUG] Error in In Vivo cumulative top metabolites callback: {e}")
            return create_empty_figure("Error", str(e))









