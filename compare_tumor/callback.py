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
from compare_tumor.data_functions import get_mz_values, get_case_columns_query, get_case_columns_vs_query, vs_columnNames, add_comparison_lines, get_case_columns_linear_query, get_cecum_and_ascending_mz_values, get_q05_mz_values, selected_mz_cleaning, get_dropdown_options, forest_plot,forest_plot_rcc_lcc, get_one_qfdr_value, get_all_columns_data, get_all_columns_data_all_compounds
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
        Output('gmm-scatter-plot', 'figure'),
        Input("selected-compound-gmm", "value") 
    )
    def gmm(selected_compound):
        print(f"Triggered callback with selected_compound: {selected_compound}")  # Debug log
        logging.info(f"Triggered callback with selected_compound: {selected_compound}")
        
        table_name = "gmm_test_1"
        logging.info("Starting to fetch data for scatter plot from the table: %s", table_name)
        print(f"Fetching data from table: {table_name}")  # Debug log

        # Fetch data
        try:
            df = get_all_columns_data(table_name, selected_compound)
            if df is None or df.empty:
                logging.warning("DataFrame is empty or not fetched from table: %s", table_name)
                return go.Figure()  # Return empty plot
            else:
                logging.info(f"Fetched {len(df)} rows from the table.")  # Log the row count
                print(f"Fetched {len(df)} rows.")  # Debug log
        except Exception as e:
            logging.error("Error fetching data: %s", e)
            return go.Figure()

        # Ensure selected compound is valid
        if not selected_compound or 'name' not in df.columns:
            logging.warning("Selected compound is invalid or 'name' column not found.")
            print("Invalid compound or missing 'name' column.")  # Debug log
            return go.Figure()

        # Reshape DataFrame for plotting
        try:
            # Filter out 'name' column from X-axis values
            columns_to_plot = [col for col in df.columns if col != 'name']
            melted_df = df.melt(id_vars=['name'], value_vars=columns_to_plot, 
                                var_name='Column', value_name='Value')
            logging.info("DataFrame melted successfully for scatter plot.")
            # print(f"Melted DataFrame:\n{melted_df.head()}")  # Debug log
        except Exception as e:
            logging.error("Error reshaping DataFrame: %s", e)
            print(f"Error reshaping DataFrame: {e}")  # Debug log
            return go.Figure()
        
        try:
            melted_df['Column'] = melted_df['Column'].str.replace("_", " ").str.upper()
            print(melted_df['Column'])
        except Exception as e:
            logging.error("Error processing 'Column' values: %s", e)
            print(f"Error processing 'Column' values: {e}")  

        fig = go.Figure()
        marker_size = max(5, 200 // len(melted_df['Column'].unique()))
        # print(f"Fetched columns {len(melted_df['Column'])} rows.")
        fig.add_trace(go.Scatter(
            x=melted_df['Column'],  # Directly pass the 'Column' series
            y=melted_df['Value'],  # Directly pass the 'Value' series
            mode='markers',
            marker=dict(size=marker_size, color='#1D78B4')
        ))

        logging.info("Scatter plot created successfully.")
        # print("Scatter plot created successfully.")  # Debug log

        # Update layout
        fig.update_layout(
            title=f'Scatter Plot for Selected Compound: {selected_compound}',
            xaxis_title='Columns',
            yaxis_title='Values',
            template="none",  # For general styling, can be set to 'none' for a plain look
            xaxis=dict(
                tickvals=list(range(len(melted_df['Column']))),
                tickfont=dict(size=max(8, 400 // len(melted_df['Column'].unique()))),
                automargin=True,      
                minor=dict(ticks='outside'),
                ticks='outside',
                ticklen=5,
                anchor='y',
                range=[0, len(melted_df['Column']) ]
 
                
            ),
            yaxis=dict(
                tickfont=dict(color='black'),
                showline=True,  # Ensures the axis line is visible
                linecolor='black',  # Make the axis line prominent
                linewidth=0.1,
                automargin=True,
                minor=dict(ticks='outside'),
                ticks='outside',
                ticklen=5,  
            ),
            
            margin=dict(b=200),  # Add more bottom margin
            plot_bgcolor="white",  # Set the background color of the plot to white
            paper_bgcolor="white"  # Set the background color of the paper (overall canvas)
        ),

        return fig



    @app.callback(
        Output('gmm-heatmap-plot', 'figure'),
        [
            Input("selected-metabolites", "value"),
            Input("selected-bacteria", "value"),
        ]
    )
    def gmm_heatmap_multiple(selected_metabolites, selected_bacteria):
        print(f"Selected Metabolites: {selected_metabolites}, Selected Bacteria: {selected_bacteria}")  # Debug log
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

        print('First DataFrame:', df)
        
        # Filter data based on selected metabolites and bacteria
        try:
            if selected_metabolites:
                df = df[["name"] + selected_metabolites]
            if selected_bacteria:
                df = df[df["name"].isin(selected_bacteria)]
            logging.info("Data filtered for heatmap.")
            print(f"Filtered DataFrame:\n{df.head()}")  # Debug log
        except Exception as e:
            logging.error("Error filtering data: %s", e)
            return go.Figure()

        try:
            # Convert numeric columns to numeric and set "name" as index
            numeric_df = df.set_index("name").apply(pd.to_numeric, errors="coerce")
            print('numeric_df', numeric_df)
            # Check for NaN values in the DataFrame
            if numeric_df.isnull().values.any():
                logging.warning("DataFrame contains NaN values. These will be ignored during summation.")
                print(f"DataFrame with NaN values:\n{numeric_df}")

            # Compute the sum of all bacteria for each metabolite (row-wise sum)
            sum_row = numeric_df.sum(axis=1, skipna=True)  

            # Add the sum row as "Net Balance"
            numeric_df["Net Balance"] = sum_row
            numeric_df = pd.concat([numeric_df[["Net Balance"]].T, numeric_df.drop(columns="Net Balance").T]).T
            print('numeric_df1', numeric_df)
            logging.info("Added row for the sum of selected bacteria.")
            print(f"Updated DataFrame with Net Balance row:\n{numeric_df.tail()}")  # Debug log
        except Exception as e:
            logging.error("Error adding Net Balance row: %s", e)
            return go.Figure()


        # Prepare heatmap data
        try:
            heatmap_data = numeric_df.T  # Transpose the DataFrame so metabolites are columns (x-axis), bacteria are rows (y-axis)
            print(f"Heatmap Data:\n{heatmap_data.head()}")  # Debug log
            
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
                margin=dict(l=100, r=100, t=50, b=250),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            return fig

        except Exception as e:
            logging.error("Error creating heatmap: %s", e)
            return go.Figure()

























    @app.callback(
        [Output(f'scatter-plot-mz_plus_h-{i}', 'figure') for i in range(7)],
        [Input('compound-dropdown-mz-plus', 'value')]
    )
    def tumor_vs_normal_plots(selected_compound):
        if selected_compound is not None:
            # Fetch and process data based on selected values
            # Assuming you have a column named "mz" in your tables
            selected_mz = float(selected_compound)

            figures = []

            for i in range(len(region)):
                # Fetch data from the database
                query_case, query_control, final_get_side_val = get_case_columns_query(
                    region[i]+"_m_plus_h", selected_mz)
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

    @app.callback(
        Output("compound-dropdown-meta", "options"),
        Output("compound-dropdown-meta", "value"),
        Input("filter-radio", "value")
    )
    def update_compound_dropdown(filter_value):
        # Logic to update options based on the selected filter

        if filter_value == "all":
            options = [{"label": mz, "value": mz}
                       for mz in get_mz_values("ascending_metabolites")]
            default_value = get_mz_values("ascending_metabolites")[0]

        elif filter_value == "across_all":
            options = [{"label": mz, "value": mz}
                       for mz in sorted(list(get_cecum_and_ascending_mz_values(["cecum_metabolites", "ascending_metabolites", "transverse_metabolites", "descending_metabolites", "sigmoid_metabolites", "rectosigmoid_metabolites", "rectum_metabolites"])), key=lambda s: str(s).casefold() if isinstance(s, str) else s)]
            default_value = sorted(list(get_cecum_and_ascending_mz_values(
                ["cecum_metabolites", "ascending_metabolites", "transverse_metabolites", "descending_metabolites", "sigmoid_metabolites", "rectosigmoid_metabolites", "rectum_metabolites"])), key=lambda s: str(s).casefold() if isinstance(s, str) else s)[0]

        elif filter_value == "specific_subsites":
            # List of all regions
            all_regions = ["cecum_metabolites", "ascending_metabolites", "transverse_metabolites",
                           "descending_metabolites", "sigmoid_metabolites", "rectosigmoid_metabolites", "rectum_metabolites"]
            options, default_value = get_one_qfdr_value(all_regions)

        elif filter_value == "proximal_distal":
            regions = ["ascending_metabolites", "cecum_metabolites", "descending_metabolites", "sigmoid_metabolites", "transverse_metabolites",
                       "rectosigmoid_metabolites", "rectum_metabolites"]
            map_region = {}
            for i in regions:
                map_region[i] = get_q05_mz_values(i)
            # print("map_region", map_region)

            # Display metabolites with q < 0.05 in cecum and ascending only and not in others
            cecum_ascending_mz_values = sorted(list(
                set(map_region["cecum_metabolites"]) & set(
                    map_region["ascending_metabolites"])
                - set(map_region["descending_metabolites"]) -
                set(map_region["sigmoid_metabolites"])
                - set(map_region["rectosigmoid_metabolites"]) - set(
                    map_region["rectum_metabolites"]) - set(map_region["transverse_metabolites"])
            ), key=lambda s: str(s).casefold() if isinstance(s, str) else s)

            # Display metabolites with q < 0.05 in descending, sigmoid, rectosigmoid, and rectum only and not in others
            descending_mz_values = sorted(list(
                set(map_region["descending_metabolites"]) & set(
                    map_region["sigmoid_metabolites"])
                & set(map_region["rectosigmoid_metabolites"]) & set(map_region["rectum_metabolites"])
                - set(map_region["cecum_metabolites"]) - set(
                    map_region["ascending_metabolites"]) - set(map_region["transverse_metabolites"])
            ), key=lambda s: str(s).casefold() if isinstance(s, str) else s)

            # Display metabolites with q < 0.05 in sigmoid, rectosigmoid, and rectum only and not in others
            sigmoid_recto_rectum_mz_values = sorted(list(
                set(map_region["sigmoid_metabolites"]) & set(
                    map_region["rectosigmoid_metabolites"]) & set(map_region["rectum_metabolites"])
                - set(map_region["cecum_metabolites"]) -
                set(map_region["ascending_metabolites"])
                - set(map_region["descending_metabolites"]) -
                set(map_region["transverse_metabolites"])
            ), key=lambda s: str(s).casefold() if isinstance(s, str) else s)

            # print("type", type(descending_mz_values))

            filter_list = []
            cecum_ascending_mz_values.extend(set(descending_mz_values))
            cecum_ascending_mz_values.extend(
                set(sigmoid_recto_rectum_mz_values))

            # print("filter_list", cecum_ascending_mz_values)
            options = [
                {"label": mz, "value": mz} for mz in cecum_ascending_mz_values
                # for mz_list in filter_list
            ]
            default_value = list(cecum_ascending_mz_values)[
                0] if cecum_ascending_mz_values else None
            # print("options1", options)
            # print("default_value1", default_value)

        else:
            # Default options and value
            options = []
            default_value = None

        return options, default_value

    @app.callback(
        [Output(f'scatter-plot-meta-{i}', 'figure') for i in range(7)],
        [Input('compound-dropdown-meta', 'value')]
    )
    def tumor_vs_normal_meta_plots(selected_compound):
        if selected_compound is not None:
            # Fetch and process data based on selected values
            # Assuming you have a column named "mz" in your tables
            selected_mz = selected_mz_cleaning(selected_compound)

            figures = []

            for i in range(len(region)):
                # Fetch data from the database
                # print("meta_valyes")
                query_case, query_control, final_get_side_val = get_case_columns_query(
                    region[i]+"_metabolites", selected_mz)
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

    @app.callback(
        Output('tumor-plus-plot', 'figure'),
        Output('normal-plus-plot', 'figure'),
        [Input('compound-dropdown-mz-plus', 'value')]
    )
    def tumor_normal_plot(selected_compound):
        if selected_compound is not None:
            # Fetch and process data based on selected values
            selected_mz = float(selected_compound)
            query_tumor_regions = []
            query_normal_regions = []

            for i in range(len(region)):
                query_case, query_control, final_get_side_val = get_case_columns_query(
                    region[i]+"_m_plus_h", selected_mz)
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
        Output('tumor-comparable-plot', 'figure'),
        Output('normal-comparable-plot', 'figure'),
        [Input('compound-dropdown-compare', 'value')]
    )
    def tumor_normal_comparable_plot(selected_compound):
        if selected_compound is not None:
            # Fetch and process data based on selected values
            selected_meta = selected_mz_cleaning(selected_compound)
            # table_name = "tumor_comparable_plots"
            query_tumor_regions = []
            query_normal_regions = []
            # vs_columnNames(selected_meta)
            # Define a list of colors for each region

            for i in region:
                # print(i)
                # print('\n')

                query_case = get_case_columns_vs_query(
                    i, selected_meta, "tumor_comparable_plots")
                query_case = list(query_case[0])
                query_tumor_regions.extend(query_case)
                # print("123qwe", query_tumor_regions)

                query_control = get_case_columns_vs_query(
                    i, selected_meta, "normal_comparable_plots")
                query_control = list(query_control[0])
                query_normal_regions.extend(query_control)
                # print("456qwe", query_normal_regions)

            tumor_plot_comparable_all_regions = make_subplots()
            tumor_plot_comparable_all_regions = comparable_plots(
                tumor_plot_comparable_all_regions, query_tumor_regions, "Tumor", "tumor_comparable_plots", selected_meta, region,'Relative Abundance(log)')

            normal_plot_comparable_all_regions = make_subplots()
            normal_plot_comparable_all_regions = comparable_plots(
                normal_plot_comparable_all_regions, query_normal_regions, "Normal", "normal_comparable_plots", selected_meta, region, 'Relative Abundance(log)')

            # Show the graph containers
            return tumor_plot_comparable_all_regions, normal_plot_comparable_all_regions
        else:
            # If dropdown is not selected, hide the containers
            return go.Figure(), go.Figure()

    # @app.callback(Output("selected-image", "src"),
    #               [Input("image-dropdown", "value")])
    # def update_selected_image(selected_value):
    #     if selected_value is not None:
    #         return selected_value
    #     else:
    #         return "assets/images/car.jpg"

    @app.callback(
        Output('tumor-comparable-rcc-lcc-plot', 'figure'),
        Output('normal-comparable-rcc-lcc-plot', 'figure'),
        [Input('compound-dropdown-compare-rcc-lcc', 'value')]
    )
    def tumor_normal_comparable_rcc_lcc_plot(selected_compound):
        if selected_compound is not None:
            # Fetch and process data based on selected values
            selected_meta = selected_mz_cleaning(selected_compound)
            # table_name = "tumor_comparable_plots"
            query_tumor_regions = []
            query_normal_regions = []
            # vs_columnNames(selected_meta)
            # Define a list of colors for each region

            region_rcc_lcc = ["rcc", "lcc", "rectum"]

            for i in region_rcc_lcc:
                # print(i)
                # print('\n')

                query_case = get_case_columns_vs_query(
                    i, selected_meta, "tumor_rcc_lcc_comparable_plots")
                query_case = list(query_case[0])
                query_tumor_regions.extend(query_case)
                # print("query_tumor_regions2", query_tumor_regions)

                query_control = get_case_columns_vs_query(
                    i, selected_meta, "normal_rcc_lcc_comparable_plots")
                query_control = list(query_control[0])
                query_normal_regions.extend(query_control)
                # print("query_normal_regions2", query_normal_regions)

            tumor_plot_comparable_all_regions = make_subplots()
            tumor_plot_comparable_all_regions = comparable_plots(
                tumor_plot_comparable_all_regions, query_tumor_regions, "Tumor", "tumor_rcc_lcc_comparable_plots", selected_meta, region_rcc_lcc, 'Relative Abundance(log)')

            normal_plot_comparable_all_regions = make_subplots()
            normal_plot_comparable_all_regions = comparable_plots(
                normal_plot_comparable_all_regions, query_normal_regions, "Normal", "normal_rcc_lcc_comparable_plots", selected_meta, region_rcc_lcc, 'Relative Abundance(log)')

            # Show the graph containers
            return tumor_plot_comparable_all_regions, normal_plot_comparable_all_regions
        else:
            # If dropdown is not selected, hide the containers
            return go.Figure(), go.Figure()

    @app.callback(
        Output('tumor-linear-plot', 'figure'),
        Output('normal-linear-plot', 'figure'),
        [Input('compound-dropdown-linear', 'value')]
    )
    def tumor_normal_linear_plot(selected_compound):
        if selected_compound is not None:
            # Fetch and process data based on selected values
            selected_meta = selected_mz_cleaning(selected_compound)
            # table_name = "tumor_comparable_plots"
            query_tumor_linear_regions = []
            query_normal_linear_regions = []
            # vs_columnNames(selected_meta)
            # Define a list of colors for each region

            for i in region:
                # print(i)
                # print('\n')

                query_case, q_fdr_case = get_case_columns_linear_query(
                    i, selected_meta, "tumor_linear_plots")
                query_case = list(query_case[0])
                query_tumor_linear_regions.extend(query_case)
                # print(query_tumor_linear_regions)

                query_control, q_fdr_control = get_case_columns_linear_query(
                    i, selected_meta, "normal_linear_plots")

                # print("q_fdr_control", q_fdr_control[0][0])
                query_control = list(query_control[0])
                query_normal_linear_regions.extend(query_control)
                # print(query_normal_linear_regions)

            tumor_linear_plot_all_regions = make_subplots()
            tumor_linear_plot_all_regions = all_regions_plots(
                tumor_linear_plot_all_regions, query_tumor_linear_regions, "Tumor", 'Relative Abundance(log)')
            qFdrStars = ''
            if q_fdr_case[0][0] <= 0.001:
                qFdrStars = '***'
                tumor_linear_plot_all_regions = addAnotations(
                    tumor_linear_plot_all_regions, qFdrStars)
            elif q_fdr_case[0][0] <= 0.01 and q_fdr_case[0][0] > 0.001:
                qFdrStars = '**'
                tumor_linear_plot_all_regions = addAnotations(
                    tumor_linear_plot_all_regions, qFdrStars)
            elif q_fdr_case[0][0] <= 0.05 and q_fdr_case[0][0] > 0.01:
                qFdrStars = '*'
                tumor_linear_plot_all_regions = addAnotations(
                    tumor_linear_plot_all_regions, qFdrStars)

            normal_linear_plot_all_regions = make_subplots()
            normal_linear_plot_all_regions = all_regions_plots(
                normal_linear_plot_all_regions, query_normal_linear_regions, "Normal", 'Relative Abundance(log)')
            qFdrStars1 = ''
            if q_fdr_control[0][0] <= 0.001:
                qFdrStars1 = '***'
                normal_linear_plot_all_regions = addAnotations(
                    normal_linear_plot_all_regions, qFdrStars1)
            elif q_fdr_control[0][0] <= 0.01 and q_fdr_control[0][0] > 0.001:
                qFdrStars1 = '**'
                normal_linear_plot_all_regions = addAnotations(
                    normal_linear_plot_all_regions, qFdrStars1)
            elif q_fdr_control[0][0] <= 0.05 and q_fdr_control[0][0] > 0.01:
                qFdrStars1 = '*'

                normal_linear_plot_all_regions = addAnotations(
                    normal_linear_plot_all_regions, qFdrStars1)

            # Show the graph containers
            return tumor_linear_plot_all_regions, normal_linear_plot_all_regions
        else:
            # If dropdown is not selected, hide the containers
            return go.Figure(), go.Figure()

    @app.callback(
        Output('forest-plot-image', 'src'),
        [Input('compound-dropdown-forest', 'value')]
    )
    def update_forest_plot(selected_mz):
        regions = ['Cecum', 'Ascending', 'Transverse',
                   'Descending', 'Sigmoid', 'Rectosigmoid', 'Rectum']
        result_list = forest_plot(selected_mz, regions)
        result_df = pd.DataFrame(result_list)

        # Create a new figure and axes
        fig, ax = plt.subplots(figsize=(6.5, 5),)
        fp.forestplot(
            result_df,
            estimate="HR",
            ll="Low",
            hl="High",
            varlabel="region",
            # ylabel="HR 95%(CI)",
            xlabel="Hazard Ratio",
            annote=["region", "est_hr"],
            annoteheaders=["          ", "HR (95%  CI)"],
            flush=False,
            ci_report=False,
            capitalize="capitalize",
            rightannote=["Pval"],
            right_annoteheaders=["P-Value"],
            table=True,
            ax=ax,
            xline_kwargs=dict(linewidth=2)
        )
        # Adjust the layout of the subplot
        plt.subplots_adjust(top=0.855, bottom=0.165, left=0.510,
                            right=0.835, hspace=0.2, wspace=0.2)

        # Save the Matplotlib figure as bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format="png",
                    bbox_inches="tight", pad_inches=0.1)
        plt.close()  # Close the Matplotlib figure to free up resources

        # Convert bytes to base64 string
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

        # Open image with PIL
        im = Image.open(io.BytesIO(img_bytes.getvalue()))

        # Get image dimensions
        width, height = im.size

        # Crop the image (20% from the left)
        new_width = int(width * 0.17)
        im1 = im.crop((new_width, 0, width, height))

        # Save the cropped image
        cropped_img_bytes = io.BytesIO()
        im1.save(cropped_img_bytes, format='PNG')
        cropped_img_base64 = base64.b64encode(
            cropped_img_bytes.getvalue()).decode('utf-8')

        # Create the image source for the cropped image
        cropped_image_src = f"data:assets/image/png;base64,{cropped_img_base64}"
        return cropped_image_src

    @app.callback(
        Output('forest-specific-plot-image', 'src'),
        [Input('compound-dropdown-forest-specific', 'value')]
    )
    def update_forest_specific_plot(selected_mz):
        regions = ['Cecum', 'Ascending', 'Transverse',
                   'Descending', 'Sigmoid', 'Rectosigmoid', 'Rectum']
        result_list = forest_plot(selected_mz, regions)
        result_df = pd.DataFrame(result_list)

        fig, ax = plt.subplots()  # Create a new figure and axes
        fp.forestplot(
            result_df,
            estimate="HR",
            ll="Low",
            hl="High",
            varlabel="region",
            # ylabel="HR 95%(CI)",
            xlabel="Hazard Ratio",
            annote=["region", "est_hr"],
            annoteheaders=["          ", "HR (95%  CI)"],
            flush=False,
            ci_report=False,
            capitalize="capitalize",
            rightannote=["Pval"],
            right_annoteheaders=["P-Value"],
            table=True,
            ax=ax,
            xline_kwargs=dict(linewidth=2)
        )
        # Adjust the layout of the subplot
        plt.subplots_adjust(top=0.855, bottom=0.165, left=0.510,
                            right=0.835, hspace=0.2, wspace=0.2)

        # Save the Matplotlib figure as bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format="png",
                    bbox_inches="tight", pad_inches=0.1)
        plt.close()  # Close the Matplotlib figure to free up resources

        # Convert bytes to base64 string
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

        # Open image with PIL
        im = Image.open(io.BytesIO(img_bytes.getvalue()))

        # Get image dimensions
        width, height = im.size

        # Crop the image (20% from the left)
        new_width = int(width * 0.17)
        im1 = im.crop((new_width, 0, width, height))

        # Save the cropped image
        cropped_img_bytes = io.BytesIO()
        im1.save(cropped_img_bytes, format='PNG')
        cropped_img_base64 = base64.b64encode(
            cropped_img_bytes.getvalue()).decode('utf-8')

        # Create the image source for the cropped image
        cropped_image_src = f"data:assets/image/png;base64,{cropped_img_base64}"
        return cropped_image_src

    @app.callback(
        Output('forest-rcc-lcc-plot-image', 'src'),
        [Input('compound-dropdown-forest-rcc-lcc', 'value')]
    )
    def update_forest_lcc_rcc_specific_plot(selected_mz):
        regions = ['RCC','LCC ','Rectum']
        result_list = forest_plot_rcc_lcc(selected_mz, regions)
        result_df = pd.DataFrame(result_list)

        fig, ax = plt.subplots()  # Create a new figure and axes
        fp.forestplot(
            result_df,
            estimate="HR",
            ll="Low",
            hl="High",
            varlabel="region",
            # ylabel="HR 95%(CI)",
            xlabel="Hazard Ratio",
            annote=["region", "est_hr"],
            annoteheaders=["          ", "HR (95%  CI)"],
            flush=False,
            ci_report=False,
            capitalize="capitalize",
            rightannote=["Pval"],
            right_annoteheaders=["P-Value"],
            table=True,
            ax=ax,
            xline_kwargs=dict(linewidth=2)
        )
        plt.ylim(-0.2884800000000002, 3.01652)
        # Adjust the layout of the subplot
        plt.subplots_adjust(top=0.735, bottom=0.230, left=0.450,
                            right=0.850, hspace=0.2, wspace=0.2)

        # Save the Matplotlib figure as bytes
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format="png",
                    bbox_inches="tight", pad_inches=0.1)
        plt.close()  # Close the Matplotlib figure to free up resources

        # Convert bytes to base64 string
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

        # Open image with PIL
        im = Image.open(io.BytesIO(img_bytes.getvalue()))

        # Get image dimensions
        width, height = im.size

        # Crop the image (20% from the left)
        new_width = int(width * 0.129)
        im1 = im.crop((new_width, 0, width, height))

        # Save the cropped image
        cropped_img_bytes = io.BytesIO()
        im1.save(cropped_img_bytes, format='PNG')
        cropped_img_base64 = base64.b64encode(
            cropped_img_bytes.getvalue()).decode('utf-8')

        # Create the image source for the cropped image
        cropped_image_src = f"data:assets/image/png;base64,{cropped_img_base64}"
        return cropped_image_src
