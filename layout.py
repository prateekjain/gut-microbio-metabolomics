# layouts.py
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from compare_tumor.callback import register_callbacks
from compare_tumor.data_functions import *
    
    
# ===== CENTRALIZED LOADING FUNCTION =====
def create_loading_wrapper(component_id, children, loading_type="circle", className=""):
    """
    Centralized loading wrapper function for consistent UX across the application.
    
    Args:
        component_id (str): Unique ID for the loading component
        children (list): Content to wrap with loading
        loading_type (str): Type of loading animation ('circle', 'dot', 'cube', 'graph', 'default')
        className (str): Additional CSS classes for the loading wrapper
    
    Returns:
        dcc.Loading: Loading component with consistent styling
    """
    return dcc.Loading(
        id=f"loading-{component_id}",
        type=loading_type,
        children=children,
        color="#667eea",
        className=f"loading-wrapper {className}",
        loading_state={'is_loading': False},
        style={
            "minHeight": "200px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center"
        }
    )

region = [
    "cecum",
    "ascending",
    "transverse",
    "descending",
    "sigmoid",
    "rectosigmoid",
    "rectum",
]


def _about_section():
    counts = get_in_vivo_feature_counts()
    hilic = f"{counts['hilic']:,}"
    rplc = f"{counts['rplc']:,}"
    return html.Div(
        className="para",
        children=[
            html.P(
                [
                    "This database is a comprehensive resource for exploring the metabolic activities of individual gut bacterial species. It integrates data from 310 ",
                    html.I("in vitro"),
                    " bacterial cultures and 111 mono-colonized mouse models. All data is visualized as log₂ fold change relative to the appropriate control groups to isolate microbial impact:",
                ],
            ),
            html.Ul(
                [
                    html.Li([html.I("In Vitro"), " data is normalized to respective blank media to reflect intrinsic microbial capacity."]),
                    html.Li([html.I("In Vivo"), " data is normalized to germ-free (GF) controls to identify specific metabolic shifts within the host."]),
                ]
            ),
            html.P(
                [
                    "By offering detailed datasets and user-friendly tools, we aim to advance the understanding of gut microbial metabolites and their interactions with the host. This resource encompasses 354 annotated metabolites ",
                    html.I("in vitro"), " (validated via authentic standards; MSI levels 1 and 2) and 453 annotated metabolites ",
                    html.I("in vivo"), " (comprising 362 MSI level 1/2 identifications and 91 ",
                    html.I("in silico"),
                    f" MSI level 3 annotations). Additionally, the resource provides data on {hilic} untargeted metabolic features acquired in HILIC ESI negative mode and {rplc} features acquired in RPLC ESI positive mode.",
                ],
            ),
        ],
    )


tabs_mz = dcc.Tabs(
    [
            dcc.Tab(
                label="Negative ions",
                value="mz-h-tab",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Select Compound:",
                                        id="mz-h-section",
                                        className="select-label",
                                    ),
                                    dcc.Dropdown(
                                        id="compound-dropdown",
                                        options=[],
                                        placeholder="Select Mz Value",
                                        searchable=True,
                                        multi=False,
                                        style={"width": "100%"},
                                        className="select-input",
                                    ),
                                    html.Div(
                                        id="selected-mz-h-value",
                                        className="select-label",
                                    ),
                                ],
                                md=12,
                            ),
                        ]
                    ),
                    dbc.Row(
                    [
                        dbc.Col(
                            [
                                create_loading_wrapper(
                                    "mz-negative-plots",
                                    [
                                        html.Div(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dcc.Graph(
                                                                    id=f"scatter-plot-mz_minus_h-{i}",
                                                                    className="scatter-plot",
                                                                )
                                                                for i in range(7)
                                                            ],
                                                            className="inner-container",
                                                        ),
                                                    ]
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dcc.Graph(
                                                                    id="tumor-plot",
                                                                    className="tumor-plot",
                                                                ),
                                                                dcc.Graph(
                                                                    id="normal-plot",
                                                                    className="normal-plot",
                                                                ),
                                                            ],
                                                            style={
                                                                "display": "flex"
                                                            },
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="outer-container with-shadow",
                                        ),
                                    ],
                                    "circle"
                                ),
                            ],
                            md=12,
                        ),
                    ]
                ),
                ],
            ),
        dcc.Tab(
            label="Positive ions",
            value="mz-plus-tab",
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label(
                                    "Select Compound:",
                                    id="mz-plus-section",
                                    className="select-label",
                                ),
                                dcc.Dropdown(
                                    id="compound-dropdown-mz-plus",
                                    options=[],
                                    placeholder="Select M+H Value",
                                    searchable=True,
                                    multi=False,
                                    style={"width": "100%"},
                                    className="select-input",
                                ),
                                html.Div(id="selected-mz-plus-value"),
                            ],
                            md=12,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                create_loading_wrapper(
                                    "mz-positive-plots",
                                    [
                                        html.Div(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dcc.Graph(
                                                                    id=f"scatter-plot-mz_plus_h-{i}",
                                                                    className="scatter-plot",
                                                                )
                                                                for i in range(7)
                                                            ],
                                                            className="inner-container",
                                                        ),
                                                    ]
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dcc.Graph(
                                                                    id="tumor-plus-plot",
                                                                    className="tumor-plot",
                                                                ),
                                                                dcc.Graph(
                                                                    id="normal-plus-plot",
                                                                    className="normal-plot",
                                                                ),
                                                            ],
                                                            style={
                                                                "display": "flex"
                                                            },
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="outer-container with-shadow",
                                        ),
                                    ],
                                    "circle"
                                ),
                            ],
                            md=12,
                        ),
                    ]
                ),
            ],
        ),
    ],
    id="tabs_mz",
    value="mz-h-tab",
    className="tabs",
)

tabs_compare = dcc.Tabs(
            [
                dcc.Tab(
                    label="7 Subsites",
                value="compare-all",
                children=[
                    dbc.Row(
                        [
                            html.Label(
                                "Select Compound:",
                                id="mz-compare-section",
                                className="select-label",
                            ),
                            dcc.Dropdown(
                                id="compound-dropdown-compare",
                                options=[],
                                placeholder="Select Mz Value",
                                searchable=True,
                                clearable=True,
                                multi=False,
                                style={"width": "100%"},
                                className="select-input",
                            ),
                            html.Div(id="selected-mz-compare-value"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Loading(
                                                id="outer-container-loading",
                                                type="circle",
                                                children=[
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                id="tumor-comparable-plot",
                                                                className="tumor-comparable-plot",
                                                            ),
                                                            dcc.Graph(
                                                                id="normal-comparable-plot",
                                                                className="normal-comparable-plot",
                                                            ),
                                                        ],
                                                        style={"display": "flex"},
                                                        className="outer-container with-shadow",
                                                    ),
                                                ],
                                            ),
                                        ],
                                        md=12,
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
            ),
            dcc.Tab(
                label="LCC, RCC, Rectum",
                value="compare-rcc-lcc",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Select Compound:",
                                        id="mz-compare-rcc-lcc-section",
                                        className="select-label",
                                    ),
                                    dcc.Dropdown(
                                        id="compound-dropdown-compare-rcc-lcc",
                                        options=[],
                                        placeholder="Select Mz Value",
                                        searchable=True,
                                        clearable=True,
                                        multi=False,
                                        style={"width": "100%"},
                                        className="select-input",
                                    ),
                                    html.Div(
                                        id="selected-mz-compare-rcc-lcc-value"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dcc.Loading(
                                                        id="outer-container-loading",
                                                        type="circle",
                                                        children=[
                                                            html.Div(
                                                                [
                                                                    dcc.Graph(
                                                                        id="tumor-comparable-rcc-lcc-plot",
                                                                        className="tumor-comparable-rcc-lcc-plot",
                                                                    ),
                                                                    dcc.Graph(
                                                                        id="normal-comparable-rcc-lcc-plot",
                                                                        className="normal-comparable-rcc-lcc-plot",
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex"},
                                                                className="outer-container with-shadow",
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                                md=12,
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
            ],
        ),
    ],
    id="tabs_compare",
    value="compare-all",
    className="tabs",
)

tabs_survival = dcc.Tabs(
    [
        dcc.Tab(
            label="All Metabolites",
            value="all-meta",
            children=[
                dbc.Col(
                    [
                        html.Label(
                            "Select Compound:",
                            id="mz-forest-section",
                            className="select-label",
                        ),
                        dcc.Dropdown(
                            id="compound-dropdown-forest",
                            options=[],
                            placeholder="Select Mz Value",
                            searchable=True,
                            multi=False,
                            style={"width": "100%"},
                            className="select-input",
                        ),
                        html.Div(id="selected-mz-forest-value"),
                    ],
                    md=12,
                ),
                dcc.Loading(
                    id="outer-container-loading",
                    type="circle",
                    children=[
                        html.Div(
                            [
                                html.Img(
                                    id='forest-plot-image',
                                    className="forest-plot",
                                    style={'width': '60%',
                                           'height': '40%',
                                           'align-item': 'center', }
                                )
                            ],
                            style={"display": "flex"},
                            className="outer-container with-shadow",
                        ),
                    ],
                ),
            ]
        ),
        dcc.Tab(
            label="Subsites specific survival markers",
            value="less-subsites",
            children=[
                dbc.Col(
                    [
                        html.Label(
                            "Select Compounds:",
                            id="mz-forest-specific-section",
                            className="select-label",
                        ),
                        dcc.Dropdown(
                            id="compound-dropdown-forest-specific",
                            options=[],
                            placeholder="Select Mz Value",
                            searchable=True,
                            multi=False,
                            style={"width": "100%"},
                            className="select-input",
                        ),
                        html.Div(id="selected-mz-forest-specific-value"),
                    ],
                    md=12,
                ),
                dcc.Loading(
                    id="outer-container-loading",
                    type="circle",
                    children=[
                        html.Div(
                            [
                                html.Img(
                                    id='forest-specific-plot-image',
                                    className="forest-plot",
                                    style={'width': '60%',
                                           'height': '40%',
                                           'align-item': 'center', }

                                )
                            ],
                            style={"display": "flex"},
                            className="outer-container with-shadow",
                        ),
                    ],
                ),
            ]
        ),
        dcc.Tab(
            label="RCC or LCC or Rectum",
            value="survival-rcc-lcc-rectum",
            children=[

                dbc.Col(
                    [
                        html.Label(
                            "Select Compound:",
                            id="mz-forest-section",
                            className="select-label",
                        ),
                        dcc.Dropdown(
                            id="compound-dropdown-forest-rcc-lcc",
                            options=[],
                            placeholder="Select Mz Value",
                            searchable=True,
                            multi=False,
                            style={"width": "100%"},
                            className="select-input",
                        ),
                        html.Div(id="selected-mz-forest-value"),
                    ],
                    md=12,
                ),
                dcc.Loading(
                    id="outer-container-loading",
                    type="circle",
                    children=[
                        html.Div(
                            [
                                html.Img(
                                    id='forest-rcc-lcc-plot-image',
                                    className="forest-plot",
                                    style={'width': '60%',
                                           'height': '40%',
                                           'align-item': 'center', }
                                )
                            ],
                            style={"display": "flex"},
                            className="outer-container with-shadow",
                        ),
                    ],
                ),
            ]
        )
    ],
    id="tabs_survival",
    value="all-meta",
    className="tabs",
)
study_info_dropdown = html.Div(
    className='dropdown',
)



footer_layout = html.Footer(
    className='footer',
    id='footer',
    
    children=[
        html.Div(className="container", children=[
            html.Div(className="footer-mega-menu", children=[
                html.Div(className="menu-section", children=[
                    html.H2("Study Information"),
                    html.Br(),
                    html.Ul([
                        html.Li(html.A("Sample cohort information", href="#cohort-popup")),
                        html.Li(html.A("Sample preparation and LC-MS Analysis", href="#preparation-popup")),
                        html.Li(html.A("Metabolite feature identification", href="#metabolite-popup")),
                        html.Li(html.A("Link to publication and citing the database", href="#citation-popup")),
                        html.Li(html.A("Project and funding information", href="#funding-popup")),
                        html.Li(html.A("Contact Us", href="#contact-popup"))
                    ])
                ]),
            ]),
            html.Div(id='cohort-popup', className='popup-overlay', children=[
                html.Div(className='popup', children=[
                    html.H2("Sample Cohort Information"),
                    html.Br(),
                    html.P("Patient-matched tumor tissues and normal mucosa tissues (collected furthest away from tumor within the subsite) were surgically removed during colectomy for colorectal cancer in the operating room at Memorial Sloan Kettering Cancer Center (MSKCC), New York, NY, USA, frozen immediately in liquid nitrogen and stored at -80oC before analysis. Sample were collected in 1991-2001. The Yale University Institutional Review Board (IRB) determined that the study conducted in this publication was not considered to be Human Subjects Research and did not require an IRB review (IRB/HSC# 1612018746). Patient characteristics can be found in supplementary table 1 in our publication: (link).",className="content"),
                    html.A("x", className='close', href="#footer"),
                ])
            ]),
            html.Div(id='preparation-popup', className='popup-overlay', children=[
                html.Div(className='popup', children=[
                    html.H2("Sample Preparation and LC-MS Analysis"),
                    html.Br(),
                    html.P("Detailed sample preparation and LC-MS information can be found in our publication here (link). The data displayed in this database was acquired from the analysis of patient-matched tumor tissues and normal mucosa using a UPLC-ESI-QTOFMS (H-Class ACQUITY and Xevo G2-XS; Waters Corporation, Milford, MA, USA) was used for MS data acquisitionby RPLC ESI positive and HILIC ESI negative mode. We chose to make our data available in the format of this database, other data requests, along with protocols and codes can be made by email, please see contact us section.", className="content"),
                    html.A("x", className='close', href="#footer"),
                ])
            ]),
            html.Div(id='metabolite-popup', className='popup-overlay', children=[
                html.Div(className='popup', children=[
                    html.H2("Metabolite Feature Identification"),
                    html.Br(),
                    html.P("In this database we have displayed all metabolite features generated from the analysis of the tumor tissues and normal mucosa tissues, by electrospray ionization (ESI) mode; negative or positive. These features are displayed in Section 1. For subsequent sections we only display annotated metabolites. The level of annotation is defined by the metabolomics standards initiative (MSI) levels; Level 1:…….Level 2:…….Level 3:……..Metabolite identification methods are published in Jain. Et al…..(paper under submission).", className="content"),
                    html.A("x", className='close', href="#footer"),
                ])
            ]),
            html.Div(id='citation-popup', className='popup-overlay', children=[
                html.Div(className='popup', children=[
                    html.H2("Link to Publication and Citing the Database"),
                    html.Br(),
                    html.P("Please cite the following: Jain A,...paper details here", className="content"),
                    html.A("x", className='close', href="#footer"),
                ])
            ]),
            html.Div(id='funding-popup', className='popup-overlay', children=[
                html.Div(className='popup', children=[
                    html.H2("Project and Funding Information"),
                    html.Br(),
                    html.P("The data acquired in this database was supported by funding from the American Cancer Society awarded to Caroline Johnson, and the Yale Center for Clinical Investigation awarded to Abhishek Jain.", className="content"),
                    html.A("x", className='close', href="#footer"),
                ])
            ]),
            html.Div(id='contact-popup', className='popup-overlay', children=[
                html.Div(className='popup', children=[
                    html.H2("Contact Us"),
                    html.Br(),
                    html.P("Please contact Caroline Johnson: caroline.johnson@yale.edu  or Abhishek Jain: a.jain@yale.edu for any inquiries.", className="content"),
                    html.A("x", className='close', href="#footer"),
                ])
            ]),
            
        ]),
        html.P("The colorectal cancer metabolome database was designed by Abhishek Jain © Johnson-lab 2024 Yale University", className="copyright"),
    ]
)
# Define your buttons
button1 = html.A(
    "Microbial Metabolic Landscape",
    id="btn-microbial-landscape",
    n_clicks=0,
    className="btn-section btn-center",
    href="#section1",
)
button2 = html.A(
    "Metabolic Interaction Networks",
    id="btn-interaction-networks",
    n_clicks=0,
    className="btn-section btn-center",
    href="#section2",
)
button3 = html.A(
    "Metabolic Co-Occurrence Analysis",
    id="btn-cooccurrence-analysis",
    n_clicks=0,
    className="btn-section btn-center",
    href="#section3",
)
# Put buttons in a table
button_table = html.Table(
    [
        html.Tr(
            [
                html.Td(button1),
                html.Td(button2),
                html.Td(button3),
            ]
        ),],
    className="table-container"
)

google_analytics_scripts = html.Div([
    html.Script(**{"async": True}, src="https://www.googletagmanager.com/gtag/js?id=G-W6VVKGXT93"),
    html.Script("""
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-W6VVKGXT93');
    """)
])

main_layout = dbc.Container(
    [
        google_analytics_scripts,
        html.Div(
            className='header-bar',
            children=[
                dbc.Row(
                    [
                        html.Div(
                            className="tab",
                            children=[
                                html.A(
                                    "More Information",
                                    className="moreinfo",
                                    href="#footer",
                                )
                            ],
                        ),
                    ]
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Br(),
                        html.H1(
                            "MetaNexus: Connecting Microbes to Metabolites",
                            className="title"
                        ),
                        html.P("About", className="about-text"),
                        _about_section(),
                        button_table,
                        html.Div(className="border-line"),
                    ],
                    md=12,
                ),
            ]
        ),
        html.Div(
            className="section-spacing",
            children=[
                html.H2(
                    "Microbial Metabolic Landscape",
                    className="section-heading",
                    id="section1",
                ),
                html.H3(
                    [html.I("In Vitro"), " vs. ", html.I("In Vivo"), " Comparison"],
                    className="section-subheading",
                ),
                html.P(
                    "Explore the metabolic profiles of gut microbial strains across experimental environments. This interface supports dual-directional querying and comparative analysis.",
                    className="section-description",
                ),
                html.Div(
                    className="feature-cards",
                    children=[
                        html.Div(
                            className="feature-card",
                            children=[
                                html.Span("01", className="feature-card__index"),
                                html.H4("Query by Metabolite", className="feature-card__label"),
                                html.P(
                                    "Identify which species produce or consume a specific compound.",
                                    className="feature-card__body",
                                ),
                            ],
                        ),
                        html.Div(
                            className="feature-card",
                            children=[
                                html.Span("02", className="feature-card__index"),
                                html.H4("Query by Species", className="feature-card__label"),
                                html.P(
                                    "Characterize the entire metabolic profile of a selected bacterium.",
                                    className="feature-card__body",
                                ),
                            ],
                        ),
                        html.Div(
                            className="feature-card",
                            children=[
                                html.Span("03", className="feature-card__index"),
                                html.H4("Advanced Filtering", className="feature-card__label"),
                                html.P(
                                    "Use the “Top/Bottom 10” toggle to isolate the most significant species for a metabolite, or the most altered metabolites for a species.",
                                    className="feature-card__body",
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tabs([
                    dcc.Tab(
                        label="In Vitro",
                        value="tab-a",
                        children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Label("Select Metabolite:", className="select-label"),
                                        create_loading_wrapper(
                                            "metabolite-dropdown-a",
                                            [
                                                dcc.Dropdown(
                                                    id="selected-metabolite-gmm-a",
                                                    options=[],
                                                    placeholder="Select Metabolite",
                                                    searchable=True,
                                                    clearable=True,
                                                    multi=False,
                                                    style={"width": "100%"},
                                                    className="select-input",
                                                ),
                                            ],
                                            "dot",
                                            "dropdown-loading"
                                        ),
                                        html.Div("— OR —", className="or-divider"),
                                        html.Label("Select Bacteria:", className="select-label"),
                                        dcc.Dropdown(
                                            id="selected-bacteria-gmm-a",
                                            options=[],
                                            placeholder="Select Bacteria",
                                            searchable=True,
                                            clearable=True,
                                            multi=False,
                                            style={"width": "100%"},
                                            className="select-input",
                                        ),
                                    ], className="step-card"),
                                    html.Div([
                                        html.Label("Filter Metabolites by Type:", className="select-label"),
                                        dbc.RadioItems(
                                            id="type-filter-radio-a",
                                            options=[
                                                {"label": "Annotated metabolites", "value": "by_name"},
                                                {"label": "Positive ions", "value": "by_positive"},
                                                {"label": "Negative ions", "value": "by_negative"},
                                                {"label": "All Types", "value": "all"}
                                            ],
                                            value="all",
                                            inline=True,
                                            className="select-input radio-horizontal",
                                        ),
                                        html.Label("Show:", className="select-label"),
                                        dbc.RadioItems(
                                            id="top-bottom-radio-a",
                                            options=[
                                                {"label": "All", "value": "all"},
                                                {"label": "Top 10", "value": "top"},
                                                {"label": "Bottom 10", "value": "bottom"}
                                            ],
                                            value="all",
                                            inline=True,
                                            className="select-input radio-horizontal",
                                        ),
                                    ], className="step-card"),
                                    create_loading_wrapper(
                                        "gmm-scatter-a",
                                        [
                                            html.Div([
                                                html.Div(
                                                    id="selected-gmm-value-a",
                                                    className="select-label",
                                                ),
                                                html.Div(
                                                    dcc.Graph(
                                                        id='gmm-scatter-plot-a',
                                                        className="gmm-scatter-plot",
                                                    ),
                                                    className="scatter-container",
                                                ),
                                            ],
                                            className="outer-container",
                                            ),
                                        ],
                                        "circle"
                                    ),
                                ], md=12),
                            ]),
                        ]
                    ),
                    dcc.Tab(
                        label="In Vivo",
                        value="tab-b",
                        children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Label("Select Metabolite:", className="select-label"),
                                        # No dcc.Loading wrapper here: the options callback fires on
                                        # every keystroke (server-side search), and a loading overlay
                                        # would steal focus from the search input on each fire.
                                        dcc.Dropdown(
                                            id="selected-metabolite-gmm-b",
                                            options=[],
                                            placeholder="Select Metabolite",
                                            searchable=True,
                                            clearable=True,
                                            multi=False,
                                            style={"width": "100%"},
                                            className="select-input",
                                        ),
                                        html.Div("— OR —", className="or-divider"),
                                        html.Label("Select Bacteria:", className="select-label"),
                                        dcc.Dropdown(
                                            id="selected-bacteria-gmm-b",
                                            options=[],
                                            placeholder="Select Bacteria",
                                            searchable=True,
                                            clearable=True,
                                            multi=False,
                                            style={"width": "100%"},
                                            className="select-input",
                                        ),
                                    ], className="step-card"),
                                    html.Div([
                                        html.Label("Filter Metabolites by Type:", className="select-label"),
                                        dbc.RadioItems(
                                            id="type-filter-radio-b",
                                            options=[
                                                {"label": "Annotated metabolites", "value": "by_name"},
                                                {"label": "Positive ions", "value": "by_positive"},
                                                {"label": "Negative ions", "value": "by_negative"},
                                                {"label": "All Types", "value": "all"}
                                            ],
                                            value="all",
                                            inline=True,
                                            className="select-input radio-horizontal",
                                        ),
                                        html.Label("Show:", className="select-label"),
                                        dbc.RadioItems(
                                            id="top-bottom-radio-b",
                                            options=[
                                                {"label": "All", "value": "all"},
                                                {"label": "Top 10", "value": "top"},
                                                {"label": "Bottom 10", "value": "bottom"}
                                            ],
                                            value="all",
                                            inline=True,
                                            className="select-input radio-horizontal",
                                        ),
                                    ], className="step-card"),
                                    create_loading_wrapper(
                                        "gmm-scatter-b",
                                        [
                                            html.Div([
                                                html.Div(
                                                    id="selected-gmm-value-b",
                                                    className="select-label",
                                                ),
                                                html.Div(
                                                    dcc.Graph(
                                                        id='gmm-scatter-plot-b',
                                                        className="gmm-scatter-plot",
                                                    ),
                                                    className="scatter-container",
                                                ),
                                            ],
                                            className="outer-container",
                                            ),
                                        ],
                                        "circle"
                                    ),
                                ], md=12),
                            ]),
                        ]
                    ),
                ], id="section1-tabs", value="tab-a", className="tabs"),
                
                                html.Div(className="border-line"),
                
                html.Div(
                    className="section-spacing",
                    children=[
                        html.H2("Metabolic Interaction Networks", className="section-heading", id="section2"),
                        html.H3(
                            "Community Dynamics & Interaction Mapping",
                            className="section-subheading",
                        ),
                        html.P(
                            [
                                "This section provides a framework to predict community-level dynamics by visualizing multiple species and metabolites simultaneously. By selecting specific bacteria and metabolites, users can observe interaction patterns and functional overlaps to infer the net metabolic balance of microbial consortia. This mapping identifies how individual production and consumption profiles combine to influence overall metabolite levels, with a toggle available to compare these behaviors between ",
                                html.I("In Vitro"), " and ", html.I("In Vivo"), " environments.",
                            ],
                            className="section-description",
                        ),
                        
                        dcc.Tabs([
                    dcc.Tab(
                        label="In Vitro",
                        value="heatmap-tab-a",
                        children=[
                            html.Div(
                                className="form-section",
                                children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Filter by Type:", className="select-label"),
                                        dbc.RadioItems(
                                            id="type-filter-radio-heatmap",
                                            options=[
                                                {"label": "Annotated metabolites", "value": "by_name"},
                                                {"label": "Positive ions", "value": "by_positive"},
                                                {"label": "Negative ions", "value": "by_negative"},
                                                {"label": "All Types", "value": "all"}
                                            ],
                                            value="all",
                                            inline=True,
                                            className="select-input radio-horizontal",
                                        ),
                                        html.Label("Select Bacteria:", className="select-label"),
                                        create_loading_wrapper(
                                            "bacteria-dropdown-heatmap-a",
                                            [
                                                dcc.Dropdown(
                                                    id="selected-metabolites",
                                                    options=[],
                                                    placeholder="Select Bacteria",
                                                    multi=True,  # Allow multi-selection
                                                    searchable=True,
                                                    clearable=True,
                                                    style={"width": "100%"},
                                                    className="select-input",
                                                ),
                                            ],
                                            "dot",
                                            "dropdown-loading"
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Select Metabolites:", className="select-label"),
                                        dcc.Dropdown(
                                            id="selected-bacteria",
                                            options=[],
                                            placeholder="Select Metabolites",
                                            multi=True,  # Allow multi-selection
                                            searchable=True,
                                            clearable=True,
                                            style={"width": "100%"},
                                            className="select-input",
                                        ),
                                    ],
                                    md=6,
                                ),
                            ]
                        ),
                create_loading_wrapper(
                    "gmm-heatmap",
                    [
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(
                                        id='gmm-heatmap-plot',
                                        className="gmm-heatmap-plot",
                                        config={'responsive': True},  # Enable responsiveness
                                    ),
                                    className="scatter-container-heat",  # Inner scrollable container
                                ),
                            ],
                            className="outer-container",
                        ),
                    ],
                    "circle",
                    "heatmap-loading"
                ),
                            ]
                        )
                    ]
                ),
                dcc.Tab(
                    label="In Vivo",
                    value="heatmap-tab-b",
                    children=[
                        html.Div(
                            className="form-section",
                            children=[
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Filter by Type:", className="select-label"),
                                                dbc.RadioItems(
                                                    id="type-filter-radio-heatmap-b",
                                                    options=[
                                                        {"label": "Annotated metabolites", "value": "by_name"},
                                                        {"label": "Positive ions", "value": "by_positive"},
                                                        {"label": "Negative ions", "value": "by_negative"},
                                                        {"label": "All Types", "value": "all"}
                                                    ],
                                                    value="all",
                                                    inline=True,
                                                    className="select-input radio-horizontal",
                                                ),

                                                html.Label("Select Metabolites", className="select-label"),
                                                # No dcc.Loading wrapper here: see selected-metabolite-gmm-b
                                                # for rationale (per-keystroke server search vs spinner overlay).
                                                dcc.Dropdown(
                                                    id="selected-metabolites-heatmap-b",
                                                    options=[],
                                                    placeholder="Select Metabolites for X-axis",
                                                    multi=True,
                                                    searchable=True,
                                                    clearable=True,
                                                    style={"width": "100%"},
                                                    className="select-input",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [   
                                                 html.Label("Select Bacteria:", className="select-label"),
                                                dcc.Dropdown(
                                                    id="selected-bacteria-heatmap-b",
                                                    options=[],
                                                    placeholder="Select Bacteria for Y-axis",
                                                    multi=True,  # Allow multi-selection
                                                    searchable=True,
                                                    clearable=True,
                                                    style={"width": "100%"},
                                                    className="select-input",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                    ]
                                ),
                                create_loading_wrapper(
                                    "gmm-heatmap-b",
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    dcc.Graph(
                                                        id='gmm-heatmap-plot-b',
                                                        className="gmm-heatmap-plot",
                                                        config={'responsive': True},  # Enable responsiveness
                                                    ),
                                                    className="scatter-container-heat",  # Inner scrollable container
                                                ),
                                            ],
                                            className="outer-container",
                                        ),
                                    ],
                                    "circle",
                                    "heatmap-loading"
                                ),
                            ]
                        )
                    ]
                ),
            ], id="heatmap-tabs", value="heatmap-tab-a", className="tabs"),
                    ]
                ),
                
                html.Div(className="border-line"),
                
                html.Div(
                    className="section-spacing",
                    style={"display": "none"},
                    children=[
                        html.H2("Top Metabolites Analysis", className="section-heading"),
                        html.P(
                            "Select bacteria to analyze their top metabolite producers.",
                            className="section-description",
                        ),
                        
                        dcc.Tabs([
                            dcc.Tab(
                                label="In Vitro",
                                value="top-tab-a",
                                children=[
                                    html.Div(
                                        className="form-section",
                                        children=[
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Filter by Type:", className="select-label"),
                                                            dbc.RadioItems(
                                                                id="type-filter-radio-top-a",
                                                                options=[
                                                                    {"label": "Annotated metabolites", "value": "by_name"},
                                                                    {"label": "Positive ions", "value": "by_positive"},
                                                                    {"label": "Negative ions", "value": "by_negative"},
                                                                    {"label": "All Types", "value": "all"}
                                                                ],
                                                                value="all",
                                                                inline=True,
                                                                className="select-input radio-horizontal",
                                                            ),
                                                            html.Label("Select Bacteria:", className="select-label"),
                                                            dcc.Dropdown(
                                                                id="selected-bacteria-top",
                                                                options=[],
                                                                placeholder="Select Bacteria",
                                                                multi=True,  # Allow multi-selection
                                                                searchable=True,
                                                                clearable=True,
                                                                style={"width": "100%"},
                                                                className="select-input",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ]
                                            ),
                                            dcc.Loading(
                                                id="outer-container-plus-loading-scatter-top",
                                                type="circle",
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                dcc.Graph(
                                                                    id='gmm-scatter-top-plot',
                                                                    className="gmm-scatter-top-plot",
                                                                    config={'responsive': True},  # Enable responsiveness
                                                                ),
                                                                className="scatter-container-top",  # Inner scrollable container
                                                            ),
                                                        ],
                                                        className="outer-container",
                                                    ),
                                                ],
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            dcc.Tab(
                                label="In Vivo",
                                value="top-tab-b",
                                children=[
                                    html.Div(
                                        className="form-section",
                                        children=[
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Filter by Type:", className="select-label"),
                                                            dbc.RadioItems(
                                                                id="type-filter-radio-top-b",
                                                                options=[
                                                                    {"label": "Annotated metabolites", "value": "by_name"},
                                                                    {"label": "Positive ions", "value": "by_positive"},
                                                                    {"label": "Negative ions", "value": "by_negative"},
                                                                    {"label": "All Types", "value": "all"}
                                                                ],
                                                                value="all",
                                                                inline=True,
                                                                className="select-input radio-horizontal",
                                                            ),
                                                            html.Label("Select Bacteria:", className="select-label"),
                                                            dcc.Dropdown(
                                                                id="selected-bacteria-top-b",
                                                                options=[],
                                                                placeholder="Select Bacteria",
                                                                multi=True,  # Allow multi-selection
                                                                searchable=True,
                                                                clearable=True,
                                                                style={"width": "100%"},
                                                                className="select-input",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ]
                                            ),
                                            dcc.Loading(
                                                id="outer-container-plus-loading-scatter-top-b",
                                                type="circle",
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                dcc.Graph(
                                                                    id='gmm-scatter-top-plot-b',
                                                                    className="gmm-scatter-top-plot",
                                                                    config={'responsive': True},  # Enable responsiveness
                                                                ),
                                                                className="scatter-container-top",  # Inner scrollable container
                                                            ),
                                                        ],
                                                        className="outer-container",
                                                    ),
                                                ],
                                            ),
                                        ]
                                    )
                                ]
                            ),
                        ], id="top-tabs", value="top-tab-a", className="tabs"),
                    ]
                ),

                html.Div(className="border-line", style={"display": "none"}),

                html.Div(
                    className="section-spacing",
                    children=[
                        html.H2("Metabolic Co-Occurrence Analysis", className="section-heading", id="section3"),
                        html.H3(
                            "Top Producer Consortia",
                            className="section-subheading",
                        ),
                        html.Div(
                            className="section-description",
                            children=[
                                html.P(
                                    "This section identifies groups of microbial species that frequently co-occur as the top 10 producers across multiple metabolites. By analyzing these shared patterns of metabolic dominance, users can uncover functional similarities and identify metabolic clusters that consistently drive high-level production across the entire set of analyzed strains."
                                ),
                                html.P(
                                    "Use the selectors below to explore co-occurrence patterns among specific strains—including pairs, triplets, quadruplets, or larger combinations—to determine which groups frequently appear together as primary metabolic contributors."
                                ),
                            ],
                        ),
                        
                        dcc.Tabs([
                            dcc.Tab(
                                label="In Vitro",
                                value="cumm-tab-a",
                                children=[
                                    html.Div(
                                        className="form-section",
                                        children=[
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Filter by Type:", className="select-label"),
                                                            dbc.RadioItems(
                                                                id="type-filter-radio-cum-a",
                                                                options=[
                                                                    {"label": "Annotated metabolites", "value": "by_name"},
                                                                    {"label": "Positive ions", "value": "by_positive"},
                                                                    {"label": "Negative ions", "value": "by_negative"},
                                                                    {"label": "All Types", "value": "all"}
                                                                ],
                                                                value="all",
                                                                inline=True,
                                                                className="select-input radio-horizontal",
                                                            ),
                                                            html.Label("Select Bacteria:", className="select-label"),
                                                            dcc.Dropdown(
                                                                id="selected-bacteria-cum-top",
                                                                options=[],
                                                                placeholder="Select Bacteria",
                                                                multi=True,  # Allow multi-selection
                                                                searchable=True,
                                                                clearable=True,
                                                                style={"width": "100%"},
                                                                className="select-input",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ]
                                            ),
                                            dcc.Loading(
                                                id="outer-container-plus-loading-scatter-cumm",
                                                type="circle",
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                dcc.Graph(
                                                                    id='gmm-scatter-cumm-top-plot',
                                                                    className="gmm-scatter-cumm-top-plot",
                                                                    config={'responsive': True},  # Enable responsiveness
                                                                ),
                                                                className="scatter-container-top",  # Inner scrollable container
                                                            ),
                                                        ],
                                                        className="outer-container",
                                                    ),
                                                ],
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            dcc.Tab(
                                label="In Vivo",
                                value="cumm-tab-b",
                                children=[
                                    html.Div(
                                        className="form-section",
                                        children=[
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Filter by Type:", className="select-label"),
                                                            dbc.RadioItems(
                                                                id="type-filter-radio-cum-b",
                                                                options=[
                                                                    {"label": "Annotated metabolites", "value": "by_name"},
                                                                    {"label": "Positive ions", "value": "by_positive"},
                                                                    {"label": "Negative ions", "value": "by_negative"},
                                                                    {"label": "All Types", "value": "all"}
                                                                ],
                                                                value="all",
                                                                inline=True,
                                                                className="select-input radio-horizontal",
                                                            ),
                                                            html.Label("Select Bacteria:", className="select-label"),
                                                            dcc.Dropdown(
                                                                id="selected-bacteria-cum-top-b",
                                                                options=[],
                                                                placeholder="Select Bacteria",
                                                                multi=True,  # Allow multi-selection
                                                                searchable=True,
                                                                clearable=True,
                                                                style={"width": "100%"},
                                                                className="select-input",
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ]
                                            ),
                                            dcc.Loading(
                                                id="outer-container-plus-loading-scatter-cumm-b",
                                                type="circle",
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                dcc.Graph(
                                                                    id='gmm-scatter-cumm-top-plot-b',
                                                                    className="gmm-scatter-cumm-top-plot",
                                                                    config={'responsive': True},  # Enable responsiveness
                                                                ),
                                                                className="scatter-container-top",  # Inner scrollable container
                                                            ),
                                                        ],
                                                        className="outer-container",
                                                    ),
                                                ],
                                            ),
                                        ]
                                    )
                                ]
                            ),
                        ], id="cumm-tabs", value="cumm-tab-a", className="tabs"),
                    ]
                ),
            ]
        ),

        footer_layout,
    ],
    fluid=True,
)
