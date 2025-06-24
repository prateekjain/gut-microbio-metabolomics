import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union
from compare_tumor.data_functions import vs_columnNames
from compare_tumor.constant import region_colors
import functools
import time

# Configure logging
logger = logging.getLogger(__name__)

region = ["cecum", "ascending", "transverse",
          "descending", "sigmoid", "rectosigmoid", "rectum"]

# ===== ENHANCED PLOTTING CONFIGURATION =====
class PlotConfig:
    """Centralized plot configuration for consistent styling"""
    
    # Color schemes
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff7f0e',
        'info': '#17becf',
        'tumor': 'rgba(255, 100, 100, 0.7)',
        'normal': 'rgba(100, 255, 100, 0.7)',
        'background': 'white',
        'grid': 'rgba(128, 128, 128, 0.2)',
        'border': 'black'
    }
    
    # Default layout settings
    DEFAULT_LAYOUT = {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'family': 'Arial, sans-serif', 'color': 'black'},
        'showlegend': False,
        'margin': {'l': 60, 'r': 60, 't': 80, 'b': 80}
    }
    
    # Box plot settings
    BOX_SETTINGS = {
        'boxpoints': 'all',
        'fillcolor': 'white',
        'line': {'color': 'black', 'width': 1},
        'jitter': 0.1,
        'pointpos': 0,
        'showlegend': False
    }
    
    # Axis settings
    AXIS_SETTINGS = {
        'mirror': True,
        'ticks': 'outside',
        'showline': True,
        'linecolor': 'black',
        'gridcolor': COLORS['grid'],
        'showgrid': True
    }

# ===== PERFORMANCE DECORATORS =====
def plot_performance_logger(func):
    """Log plot generation performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Plot {func.__name__} generated in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Plot {func.__name__} failed after {execution_time:.3f} seconds: {e}")
            raise
    return wrapper

def validate_data(func):
    """Validate input data before plotting"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Basic data validation
        for arg in args:
            if isinstance(arg, (list, np.ndarray)) and len(arg) == 0:
                logger.warning(f"Empty data passed to {func.__name__}")
                return create_empty_plot("No Data", "No data available for plotting")
        return func(*args, **kwargs)
    return wrapper

# ===== CORE PLOTTING UTILITIES =====
def create_empty_plot(title: str = "No Data", message: str = "No data available") -> go.Figure:
    """Create a standardized empty plot with message"""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[{
            'text': message,
            'xref': 'paper',
            'yref': 'paper',
            'x': 0.5,
            'y': 0.5,
            'showarrow': False,
            'font': {'size': 16, 'color': 'gray'},
            'xanchor': 'center',
            'yanchor': 'middle'
        }],
        **PlotConfig.DEFAULT_LAYOUT
    )
    return fig

def calculate_significance_stars(p_value: float) -> str:
    """Calculate significance stars based on p-value"""
    if p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    else:
        return 'NS'

def clean_and_validate_numeric_data(data: Union[List, np.ndarray]) -> np.ndarray:
    """Clean and validate numeric data for plotting"""
    if isinstance(data, list):
        data = np.array(data)
    
    # Remove NaN and infinite values
    data = data[np.isfinite(data)]
    
    # Convert to float64 for consistency
    return data.astype(np.float64)

def get_dynamic_plot_size(data_length: int, base_width: int = 300, base_height: int = 500) -> Tuple[int, int]:
    """Calculate dynamic plot size based on data"""
    # Adjust width based on data points
    width = max(base_width, min(800, base_width + data_length * 10))
    height = base_height
    return width, height

# ===== ENHANCED PLOTTING FUNCTIONS =====
@plot_performance_logger
@validate_data
def create_tumor_vs_normal_plot(
    tumor_data: List[float], 
    normal_data: List[float], 
    metadata: List[float], 
    region_name: str,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> go.Figure:
    """
    Enhanced tumor vs normal comparison plot with better performance and validation
    
    Args:
        tumor_data: Tumor sample values
        normal_data: Normal sample values  
        metadata: [q_fdr, log_fc] values
        region_name: Name of the region
        width, height: Optional custom dimensions
    """
    
    # Clean and validate data
    tumor_clean = clean_and_validate_numeric_data(tumor_data)
    normal_clean = clean_and_validate_numeric_data(normal_data)
    
    if len(tumor_clean) == 0 or len(normal_clean) == 0:
        return create_empty_plot("Invalid Data", f"No valid data for {region_name}")
    
    # Calculate significance
    q_fdr = metadata[0] if len(metadata) > 0 else 1.0
    log_fc = metadata[1] if len(metadata) > 1 else 0.0
    significance_stars = calculate_significance_stars(q_fdr)
    
    # Dynamic sizing
    if width is None or height is None:
        calc_width, calc_height = get_dynamic_plot_size(len(tumor_clean) + len(normal_clean))
        width = width or calc_width
        height = height or calc_height
    
    # Create figure
    fig = go.Figure()
    
    # Add tumor data
    fig.add_trace(go.Box(
        x=['Tumor'] * len(tumor_clean),
        y=tumor_clean,
        name='Tumor',
        marker_color=PlotConfig.COLORS['tumor'],
        **PlotConfig.BOX_SETTINGS
    ))
    
    # Add normal data
    fig.add_trace(go.Box(
        x=['Normal'] * len(normal_clean),
        y=normal_clean,
        name='Normal', 
        marker_color=PlotConfig.COLORS['normal'],
        **PlotConfig.BOX_SETTINGS
    ))
    
    # Update layout with enhanced styling
    fig.update_layout(
        width=width,
        height=height,
        title={
            'text': f'<b>{region_name}</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis={
            'title': 'Sample Type',
            **PlotConfig.AXIS_SETTINGS
        },
        yaxis={
            'title': 'Relative Abundance',
            **PlotConfig.AXIS_SETTINGS
        },
        annotations=[{
            'x': 0.98,
            'y': 0.98,
            'xref': 'paper',
            'yref': 'paper',
            'text': f"FDR: {significance_stars}<br>LogFC: {log_fc:.2f}",
            'align': 'right',
            'showarrow': False,
            'font': {'size': 11, 'color': 'black'},
            'bordercolor': 'black',
            'borderwidth': 1,
            'bgcolor': 'rgba(255, 255, 255, 0.8)'
        }],
        **PlotConfig.DEFAULT_LAYOUT
    )
    
    return fig

@plot_performance_logger
@validate_data
def create_multi_region_plot(
    data_by_region: Dict[str, List[float]], 
    title: str,
    plot_type: str = 'box',
    width: int = 700,
    height: int = 500
) -> go.Figure:
    """
    Enhanced multi-region comparison plot with dynamic configuration
    
    Args:
        data_by_region: Dictionary mapping region names to data arrays
        title: Plot title
        plot_type: Type of plot ('box', 'violin', 'scatter')
        width, height: Plot dimensions
    """
    
    fig = go.Figure()
    
    # Process each region
    for i, (region_name, data) in enumerate(data_by_region.items()):
        clean_data = clean_and_validate_numeric_data(data)
        
        if len(clean_data) == 0:
            continue
            
        color = region_colors.get(region_name, PlotConfig.COLORS['primary'])
        
        if plot_type == 'box':
            fig.add_trace(go.Box(
                x=[region_name] * len(clean_data),
                y=clean_data,
                name=region_name,
                marker_color=color,
                **PlotConfig.BOX_SETTINGS
            ))
        elif plot_type == 'violin':
            fig.add_trace(go.Violin(
                x=[region_name] * len(clean_data),
                y=clean_data,
                name=region_name,
                fillcolor=color,
                line_color='black',
                showlegend=False
            ))
        elif plot_type == 'scatter':
            fig.add_trace(go.Scatter(
                x=[i] * len(clean_data),
                y=clean_data,
                mode='markers',
                name=region_name,
                marker=dict(color=color, size=8, opacity=0.7),
                showlegend=False
            ))
    
    # Enhanced layout
    fig.update_layout(
        width=width,
        height=height,
        title={
            'text': f'<b>{title}</b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis={
            'title': 'Region',
            'tickangle': 45,
            **PlotConfig.AXIS_SETTINGS
        },
        yaxis={
            'title': 'Relative Abundance',
            **PlotConfig.AXIS_SETTINGS
        },
        **PlotConfig.DEFAULT_LAYOUT
    )
    
    return fig

@plot_performance_logger
def create_enhanced_scatter_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    title: str = "Scatter Plot",
    width: int = 800,
    height: int = 600
) -> go.Figure:
    """
    Create an enhanced scatter plot with optional color and size encoding
    
    Args:
        data: DataFrame with plot data
        x_col, y_col: Column names for x and y axes
        color_col: Optional column for color encoding
        size_col: Optional column for size encoding
        title: Plot title
        width, height: Plot dimensions
    """
    
    if data.empty:
        return create_empty_plot("No Data", "DataFrame is empty")
    
    fig = go.Figure()
    
    # Prepare scatter plot parameters
    scatter_params = {
        'x': data[x_col],
        'y': data[y_col],
        'mode': 'markers',
        'text': data.index if hasattr(data, 'index') else None,
        'hovertemplate': f'<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y}}<extra></extra>'
    }
    
    # Add color encoding
    if color_col and color_col in data.columns:
        scatter_params['marker'] = {
            'color': data[color_col],
            'colorscale': 'Viridis',
            'showscale': True,
            'colorbar': {'title': color_col}
        }
    
    # Add size encoding
    if size_col and size_col in data.columns:
        if 'marker' not in scatter_params:
            scatter_params['marker'] = {}
        scatter_params['marker']['size'] = data[size_col]
        scatter_params['marker']['sizemode'] = 'diameter'
        scatter_params['marker']['sizeref'] = 2. * max(data[size_col]) / (40.**2)
    
    fig.add_trace(go.Scatter(**scatter_params))
    
    # Calculate dynamic width based on data
    calc_width = max(width, len(data) * 15)
    
    fig.update_layout(
        width=min(calc_width, 1200),  # Cap at 1200px
        height=height,
        title={
            'text': f'<b>{title}</b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis={
            'title': x_col.replace('_', ' ').title(),
            **PlotConfig.AXIS_SETTINGS
        },
        yaxis={
            'title': y_col.replace('_', ' ').title(),
            **PlotConfig.AXIS_SETTINGS
        },
        **PlotConfig.DEFAULT_LAYOUT
    )
    
    return fig

@plot_performance_logger
def create_enhanced_heatmap(
    data: pd.DataFrame,
    title: str = "Heatmap",
    colorscale: str = 'RdYlGn',
    width: Optional[int] = None,
    height: Optional[int] = None
) -> go.Figure:
    """
    Create an enhanced heatmap with dynamic sizing and better formatting
    
    Args:
        data: DataFrame for heatmap
        title: Plot title
        colorscale: Color scale name
        width, height: Optional dimensions
    """
    
    if data.empty:
        return create_empty_plot("No Data", "No data available for heatmap")
    
    # Dynamic sizing based on data dimensions
    if width is None:
        width = max(600, min(1200, data.shape[1] * 60))
    if height is None:
        height = max(400, min(800, data.shape[0] * 40))
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=colorscale,
        colorbar={'title': 'Value'},
        text=data.values,
        texttemplate='%{text:.2f}',
        textfont={'size': 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        width=width,
        height=height,
        title={
            'text': f'<b>{title}</b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis={
            'title': 'Features',
            'tickangle': 45,
            'side': 'bottom'
        },
        yaxis={
            'title': 'Samples',
            'tickangle': 0
        },
        **PlotConfig.DEFAULT_LAYOUT
    )
    
    return fig

# ===== BACKWARDS COMPATIBILITY FUNCTIONS =====
# Keep old function names for backwards compatibility but use new implementations

def tumor_vs_normal_plot(query_case, query_control, final_get_side_val, region_name):
    """Backwards compatible wrapper for create_tumor_vs_normal_plot"""
    return create_tumor_vs_normal_plot(
        tumor_data=query_case,
        normal_data=query_control,
        metadata=final_get_side_val,
        region_name=region_name
    )

def all_regions_plots(plot_all_regions, query_regions, title, title_y='Relative Abundance'):
    """Enhanced backwards compatible multi-region plot"""
    
    # Convert flat list to region-based dictionary
    data_per_region = len(query_regions) // len(region)
    data_by_region = {}
    
    for i, region_name in enumerate(region):
        start_idx = i * data_per_region
        end_idx = (i + 1) * data_per_region
        data_by_region[region_name] = query_regions[start_idx:end_idx]
    
    return create_multi_region_plot(
        data_by_region=data_by_region,
        title=f"All Regions {title}"
    )

def comparable_plots(plot_all_regions, query_regions, title, table_name, selected_meta, region_call, title_y='Relative Abundance'):
    """Enhanced backwards compatible comparable plots with significance testing"""
    
    # Convert to region-based format  
    data_per_region = len(query_regions) // len(region_call)
    data_by_region = {}
    
    for i, region_name in enumerate(region_call):
        start_idx = i * data_per_region
        end_idx = (i + 1) * data_per_region
        data_by_region[region_name] = query_regions[start_idx:end_idx]
    
    fig = create_multi_region_plot(
        data_by_region=data_by_region,
        title=f"Comparison: {title}"
    )
    
    # Add significance annotations
    try:
        vs_columnNames(table_name, fig, selected_meta, region_call)
    except Exception as e:
        logger.warning(f"Failed to add significance annotations: {e}")
    
    return fig

def addAnotations(plot_all_regions, qFdrStars):
    """Enhanced annotation function"""
    plot_all_regions.update_layout(
        annotations=[{
            'x': 0.5,
            'y': 1.02,
            'xref': 'paper',
            'yref': 'paper',
            'text': f"<b>{qFdrStars}</b>",
            'align': 'center',
            'showarrow': False,
            'font': {'size': 16, 'color': 'black', 'family': 'Arial'},
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': 'black',
            'borderwidth': 1
        }]
    )
    return plot_all_regions

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
        # Create simple empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No Data",
            annotations=[{
                'text': "No data available for selection",
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 16},
                'x': 0.5,
                'y': 0.5,
            }],
            template="plotly_white",
        )
        return fig
    
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
