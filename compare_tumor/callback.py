# callback.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
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

from compare_tumor.dynamicPlots import tumor_vs_normal_plot, all_regions_plots, comparable_plots, addAnotations
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


def register_callbacks(app):
    # Callbacks to show/hide details
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
        Input("selected-bacteria-gmm", "value")]
    )
    def update_scatter_plot(selected_metabolite, selected_bacteria):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        logging.info(f"Triggered by: {triggered_id}")
        table_name = "gmm_test_1"

        try:
            # Reset other dropdown based on which was triggered
            if triggered_id == "selected-metabolite-gmm" and selected_metabolite:
                selected_bacteria = None  # Reset bacteria dropdown
                df = get_metabolite_data(table_name, selected_metabolite)
                plot_type = "metabolite"
            elif triggered_id == "selected-bacteria-gmm" and selected_bacteria:
                selected_metabolite = None  # Reset metabolite dropdown
                df = get_bacteria_data(table_name, selected_bacteria)
                plot_type = "bacteria"
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
                xaxis=dict(
                    tickangle=90,
                    tickfont=dict(size=max(12, 100 // len(x_axis.unique()))),
                    automargin=True,
                    ticks='outside',
                    ticklen=5,
                    range= [0, len(x_axis.unique())]
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
                ),
                margin=dict(
                    l=50,   # left margin
                    r=50,   # right margin
                    b=150,  # bottom margin - increased to accommodate labels
                    t=50,   # top margin
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

        table_name = "gmm_test_1"

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

            # Update layout
            fig.update_layout(
                title="Scatter Plot of Top 10 Metabolites for Selected Bacteria",
                xaxis_title="Metabolite",
                yaxis_title="Value",
                template="plotly_white",
                xaxis=dict(tickangle=90, showgrid=True),
                yaxis=dict(showgrid=True),
                legend_title="Bacteria",
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
        table_name = "gmm_test_1"

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
    
    
    @app.callback(
        Output('gmm-heatmap-plot', 'figure'),
        [
            Input("selected-metabolites", "value"),
            Input("selected-bacteria", "value"),
        ]
    )
    def gmm_heatmap_multiple(selected_metabolites, selected_bacteria):
        # print(f"Selected Metabolites: {selected_metabolites}, Selected Bacteria: {selected_bacteria}")  # Debug log
        logging.info(f"Selected Metabolites: {selected_metabolites}, Selected Bacteria: {selected_bacteria}")
        
        table_name = "gmm_test_1"

        # Fetch all data
        try:
            df = get_all_columns_data_all_compounds(table_name)
            if df is None or df.empty:
                logging.warning("DataFrame is empty or not fetched from table: %s", table_name)
                return go.Figure()  # Return empty plot
        except Exception as e:
            logging.error("Error fetching data: %s", e)
            return go.Figure()

        
        # Filter data based on selected metabolites and bacteria
        try:
            if selected_metabolites:
                df = df[["name"] + selected_metabolites]
            if selected_bacteria:
                df = df[df["name"].isin(selected_bacteria)]
            logging.info("Data filtered for heatmap.")
            # print(f"Filtered DataFrame:\n{df.head()}")  # Debug log
        except Exception as e:
            logging.error("Error filtering data: %s", e)
            return go.Figure()

        try:
            # Convert numeric columns to numeric and set "name" as index
            numeric_df = df.set_index("name").apply(pd.to_numeric, errors="coerce")
            # print('numeric_df', numeric_df)
            # Check for NaN values in the DataFrame
            if numeric_df.isnull().values.any():
                logging.warning("DataFrame contains NaN values. These will be ignored during summation.")
                # print(f"DataFrame with NaN values:\n{numeric_df}")

            # Compute the sum of all bacteria for each metabolite (row-wise sum)
            sum_row = numeric_df.sum(axis=1, skipna=True)  

            # Add the sum row as "Net Balance"
            numeric_df["Net Balance"] = sum_row
            numeric_df = pd.concat([numeric_df[["Net Balance"]].T, numeric_df.drop(columns="Net Balance").T]).T
            # print('numeric_df1', numeric_df)
            logging.info("Added row for the sum of selected bacteria.")
            # print(f"Updated DataFrame with Net Balance row:\n{numeric_df.tail()}")  # Debug log
        except Exception as e:
            logging.error("Error adding Net Balance row: %s", e)
            return go.Figure()


        # Prepare heatmap data
        try:
            heatmap_data = numeric_df.T  # Transpose the DataFrame so metabolites are columns (x-axis), bacteria are rows (y-axis)
            # print(f"Heatmap Data:\n{heatmap_data.head()}")  # Debug log
            
        except Exception as e:
            logging.error("Error preparing data for heatmap: %s", e)
            return go.Figure()

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

            # Update layout
            fig.update_layout(
                title='Heatmap for Selected Metabolites and Bacteria',
                xaxis_title='Metabolites',
                yaxis_title='Bacteria',
                xaxis=dict(tickangle=90),
                yaxis=dict(tickfont=dict(color='black')),
                margin=dict(l=100, r=100, t=50, b=50),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=max(600, len(heatmap_data.index) * 20),  # Adjust height dynamically
            )
            return fig

        except Exception as e:
            logging.error("Error creating heatmap: %s", e)
            return go.Figure()


    





















