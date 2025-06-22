# layouts.py
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from compare_tumor.callback import register_callbacks
from compare_tumor.data_functions import *
    
    

region = [
    "cecum",
    "ascending",
    "transverse",
    "descending",
    "sigmoid",
    "rectosigmoid",
    "rectum",
]


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
                                        options=[
                                            {"label": mz, "value": mz}
                                            for mz in get_mz_values("ascending")
                                        ],
                                        placeholder="Select Mz Value",
                                        searchable=True,
                                        multi=False,
                                        style={"width": "100%"},
                                        className="select-input",
                                        value=get_mz_values("ascending")[0],
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
                                dcc.Loading(
                                    id="outer-container-loading",
                                    type="circle",
                                    children=[
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
                                    options=[
                                        {"label": mz, "value": mz}
                                        for mz in get_mz_values("ascending_m_plus_h")
                                    ],
                                    placeholder="Select M+H Value",
                                    searchable=True,
                                    multi=False,
                                    style={"width": "100%"},
                                    className="select-input",
                                    value=get_mz_values(
                                        "ascending_m_plus_h")[0],
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
                                dcc.Loading(
                                    id="outer-container-plus-loading",
                                    type="circle",
                                    children=[
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
                                options=[
                                    {"label": mz, "value": mz}
                                    for mz in get_mz_values("tumor_comparable_plots")
                                ],
                                placeholder="Select Mz Value",
                                searchable=True,
                                clearable=True,
                                multi=False,
                                style={"width": "100%"},
                                className="select-input",
                                value=get_mz_values("tumor_comparable_plots")[0],
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
                                        options=[
                                            {"label": mz, "value": mz}
                                            for mz in get_mz_values("tumor_rcc_lcc_comparable_plots")

                                        ],
                                        placeholder="Select Mz Value",
                                        searchable=True,
                                        clearable=True,
                                        multi=False,
                                        style={"width": "100%"},
                                        className="select-input",
                                        value=get_mz_values(
                                            "tumor_rcc_lcc_comparable_plots")[0],
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
                            options=[
                                {"label": mz, "value": mz}
                                for mz in get_mz_values("forest_plot")
                            ],

                            placeholder="Select Mz Value",
                            searchable=True,
                            multi=False,
                            style={"width": "100%"},
                            className="select-input",
                            value=get_mz_values("forest_plot")[0],
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
                            options=[
                                {"label": mz, "value": mz}
                                for mz in get_q05_mz_forest_values()
                            ],

                            placeholder="Select Mz Value",
                            searchable=True,
                            multi=False,
                            style={"width": "100%"},
                            className="select-input",
                            value=list(get_q05_mz_forest_values())[0],
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
                            options=[
                                {"label": mz, "value": mz}
                                for mz in get_mz_values("forest_rcc_lcc_plot")
                            ],

                            placeholder="Select Mz Value",
                            searchable=True,
                            multi=False,
                            style={"width": "100%"},
                            className="select-input",
                            value=get_mz_values("forest_rcc_lcc_plot")[0],
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
    "Tumor vs Normal (Metabolite features)",
    id="btn-mz-h",
    n_clicks=0,
    className="btn-section btn-center",
    href="#section1",
)
button2 = html.A(
    "Tumor vs Normal (Annotated Metabolites)",
    id="btn-mz-Mucosa2",
    n_clicks=0,
    className="btn-section btn-center",
    href="#section2",
)
button3 = html.A(
    "Inter-subsite comparisons",
    id="btn-inter-subsite",
    n_clicks=0,
    className="btn-section btn-center",
    href="#section3",
)
button4 = html.A(
    "Concentration gradient of metabolites",
    id="btn-mz-linear",
    n_clicks=0,
    className="btn-section btn-center",
    href="#section4",
)
button5 = html.A(
    "Survival markers",
    id="btn-mz-Survival",
    n_clicks=0,
    className="btn-section btn-center",
    href="#section5",
)
# Put buttons in a table
button_table = html.Table(
    [
        html.Tr(
            [
                html.Td(button1, ),  # This cell spans 1 column
                html.Td(button2, ),  # This cell spans 1 column
                html.Td(button3, ),  # This cell spans 1 column
            ]
        ),],
    className="table-container"
)

button_table2 = html.Table([
        html.Tr(
            [
                # This cell spans 1 column
                html.Td(button4,  className="cell21"),
                html.Td(button5, className="cell22"),  # This cell spans 1 column
            ]
        ),
    ],
    className="table-container2"
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
                            "Microbiome Metabolome Database",
                            className="title"
                        ),
                        html.P("About", className="about-text"),
                        html.P(
                            [
                                "This database is a comprehensive resource for exploring the metabolic activities of individual gut bacterial species and their roles in human health and disease. "
                                "It integrates data from 310 in vitro bacterial cultures and 112 monocolonized mouse models, each hosting a single bacterial species, providing insights into species-specific metabolic profiles in both controlled environments and host systems. By comparing in vitro and in vivo metabolomics data, the database highlights key biochemical pathways and their significance in host-microbe interactions, including immune modulation and other physiological processes. ",
                                html.Br(),
                                html.Br(),
                                "By offering detailed datasets and user-friendly tools for analysis and visualization, we aim to advance the understanding of gut microbial metabolites and their interactions with the host. Discover the intricate metabolic landscape of gut bacteria with us and contribute to groundbreaking insights in microbiome research.",
                            ],
                            className="para",
                        ),
                        button_table,
                        button_table2,
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
                    "In Vitro vs In Vivo",
                    className="section-heading",
                    id="section1",
                ),
                html.P(
                    "Explore the metabolic differences between controlled laboratory conditions (in vitro) and living host environments (in vivo). Compare bacterial metabolite production across different experimental conditions.",
                    className="section-description",
                ),
                dcc.Tabs([
                    dcc.Tab(
                        label="In Vitro",
                        value="tab-a",
                        children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Filter by Type:", className="select-label"),
                                    dbc.RadioItems(
                                        id="type-filter-radio-a",
                                        options=[
                                            {"label": "By Name", "value": "by_name"},
                                            {"label": "By Positive", "value": "by_positive"},
                                            {"label": "By Negative", "value": "by_negative"},
                                            {"label": "All Types", "value": "all"}
                                        ],
                                        value="all",
                                        inline=True,
                                        className="select-input",
                                    ),
                                    html.Label("Select Metabolite:", className="select-label"),
                                    dcc.Dropdown(
                                        id="selected-metabolite-gmm-a",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in list(get_gmm_name("gmm_test_1"))
                                        ],
                                        placeholder="Select Metabolite",
                                        searchable=True,
                                        clearable=True,
                                        multi=False,
                                        style={"width": "100%"},
                                        className="select-input",
                                    ),
                                    dbc.RadioItems(
                                        id="top-bottom-radio-a",
                                        options=[
                                            {"label": "All", "value": "all"},
                                            {"label": "Top 10", "value": "top"},
                                            {"label": "Bottom 10", "value": "bottom"}
                                        ],
                                        value="all",
                                        inline=True,
                                        className="select-input",
                                    ),
                                    html.Label("Select Bacteria:", className="select-label"),
                                    dcc.Dropdown(
                                        id="selected-bacteria-gmm-a",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in list(get_column_names("gmm_test_1"))
                                        ],
                                        placeholder="Select Bacteria",
                                        searchable=True,
                                        clearable=True,
                                        multi=False,
                                        style={"width": "100%"},
                                        className="select-input",
                                    ),
                                    dcc.Loading(
                                        id="outer-container-plus-loading-a",
                                        type="circle",
                                        children=[
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
                                    html.Label("Filter by Type:", className="select-label"),
                                    dbc.RadioItems(
                                        id="type-filter-radio-b",
                                        options=[
                                            {"label": "By Name", "value": "by_name"},
                                            {"label": "By Positive", "value": "by_positive"},
                                            {"label": "By Negative", "value": "by_negative"},
                                            {"label": "All Types", "value": "all"}
                                        ],
                                        value="all",
                                        inline=True,
                                        className="select-input",
                                    ),
                                    html.Label("Select Metabolite:", className="select-label"),
                                    dcc.Dropdown(
                                        id="selected-metabolite-gmm-b",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in list(get_gmm_name("in_vivo"))
                                        ],
                                        placeholder="Select Metabolite",
                                        searchable=True,
                                        clearable=True,
                                        multi=False,
                                        style={"width": "100%"},
                                        className="select-input",
                                    ),
                                    dbc.RadioItems(
                                        id="top-bottom-radio-b",
                                        options=[
                                            {"label": "All", "value": "all"},
                                            {"label": "Top 10", "value": "top"},
                                            {"label": "Bottom 10", "value": "bottom"}
                                        ],
                                        value="all",
                                        inline=True,
                                        className="select-input",
                                    ),
                                    html.Label("Select Bacteria:", className="select-label"),
                                    dcc.Dropdown(
                                        id="selected-bacteria-gmm-b",
                                        options=[
                                            {"label": name, "value": name}
                                            for name in list(get_column_names("in_vivo"))
                                        ],
                                        placeholder="Select Bacteria",
                                        searchable=True,
                                        clearable=True,
                                        multi=False,
                                        style={"width": "100%"},
                                        className="select-input",
                                    ),
                                    dcc.Loading(
                                        id="outer-container-plus-loading-b",
                                        type="circle",
                                        children=[
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
                        html.H3("Metabolite-Bacteria Heatmap", className="section-heading"),
                        html.P(
                            "Select multiple bacteria and metabolites to visualize their relationships in an interactive heatmap.",
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
                                                {"label": "By Name", "value": "by_name"},
                                                {"label": "By Positive", "value": "by_positive"},
                                                {"label": "By Negative", "value": "by_negative"},
                                                {"label": "All Types", "value": "all"}
                                            ],
                                            value="all",
                                            inline=True,
                                            className="select-input",
                                        ),
                                        html.Label("Select Bacteria:", className="select-label"),
                                        dcc.Dropdown(
                                            id="selected-metabolites",
                                            options=[
                                                {"label": name, "value": name} for name in list(get_column_names("gmm_test_1"))
                                            ],
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
                                dbc.Col(
                                    [
                                        html.Label("Select Metabolites:", className="select-label"),
                                        dcc.Dropdown(
                                            id="selected-bacteria",
                                            options=[
                                                {"label": name, "value": name} for name in list(get_bacteria_names("gmm_test_1"))
                                            ],
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
                dcc.Loading(
                    id="outer-container-plus-loading-heatmap",
                    type="circle",
                    children=[
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
                                                        {"label": "By Name", "value": "by_name"},
                                                        {"label": "By Positive", "value": "by_positive"},
                                                        {"label": "By Negative", "value": "by_negative"},
                                                        {"label": "All Types", "value": "all"}
                                                    ],
                                                    value="all",
                                                    inline=True,
                                                    className="select-input",
                                                ),

                                                html.Label("Select Metabolites", className="select-label"),
                                                dcc.Dropdown(
                                                    id="selected-metabolites-heatmap-b",
                                                    options=[
                                                        {"label": name, "value": name} for name in list(get_gmm_name("in_vivo"))
                                                    ],
                                                    placeholder="Select Metabolites for X-axis",
                                                    multi=True,  # Allow multi-selection
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
                                                    options=[
                                                        {"label": name, "value": name} for name in list(get_column_names("in_vivo"))
                                                    ],
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
                                dcc.Loading(
                                    id="outer-container-plus-loading-heatmap-b",
                                    type="circle",
                                    children=[
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
                    children=[
                        html.H3("Top Metabolites Analysis", className="section-heading"),
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
                                                    html.Label("Select Bacteria:", className="select-label"),
                                                    dcc.Dropdown(
                                                        id="selected-bacteria-top",
                                                        options=[
                                                            {"label": name, "value": name} for name in list(get_column_names("gmm_test_1"))
                                                        ],
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
                                                    html.Label("Select Bacteria:", className="select-label"),
                                                    dcc.Dropdown(
                                                        id="selected-bacteria-top-b",
                                                        options=[
                                                            {"label": name, "value": name} for name in list(get_column_names("in_vivo"))
                                                        ],
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
                
                html.Div(className="border-line"),
                
                html.Div(
                    className="section-spacing",
                    children=[
                        html.H3("Cumulative Top Metabolites", className="section-heading"),
                html.P(
                    "Analyze cumulative metabolite production patterns across selected bacteria.",
                    className="section-description",
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
                                                    html.Label("Select Bacteria:", className="select-label"),
                                                    dcc.Dropdown(
                                                        id="selected-bacteria-cum-top",
                                                        options=[
                                                            {"label": name, "value": name} for name in list(get_column_names("gmm_test_1"))
                                                        ],
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
                                                    html.Label("Select Bacteria:", className="select-label"),
                                                    dcc.Dropdown(
                                                        id="selected-bacteria-cum-top-b",
                                                        options=[
                                                            {"label": name, "value": name} for name in list(get_column_names("in_vivo"))
                                                        ],
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
