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


# mega_menu = html.Div([
#     html.Div(className="navbar", children=[
        
#         html.Div(className="tab", children=[
#             "Study Information 1",
#             html.Div(className="dropdown-content", id="dropdown-content-1", children=[
#                 html.Div(className="options-column", children=[
#                     html.Div(className="option", children=[
#                              "Sample cohort informatio"], id="cohort-option"),
#                     html.Div(className="option", children=[
#                              "Sample preparation and LC-MS Analysis"], id="preparation-option"),
#                     html.Div(className="option", children=[
#                              "Metabolic feature identification"], id="feature-option"),
#                     html.Div(className="option", children=[
#                              "Link to publication and citing the database"], id="link-option"),
#                     html.Div(className="option", children=[
#                              "Project and funding information"], id="project-option"),
#                     html.Div(className="option", children=[
#                              "Contact"], id="contact-option")
#                 ]),
#                 html.Div(className="options-column", children=[
#                     html.Div(className="option-details", children=[
#                         "Patient-matched tumor tissues and normal mucosa tissues (collected furthest away from tumor within the subsite) were surgically removed during colectomy for colorectal cancer in the operating room at Memorial Sloan Kettering Cancer Center (MSKCC), New York, NY, USA, frozen immediately in liquid nitrogen and stored at -80oC before analysis. Sample were collected in 1991-2001. The Yale University Institutional Review Board (IRB) determined that the study conducted in this publication was not considered to be Human Subjects Research and did not require an IRB review (IRB/HSC# 1612018746). Patient characteristics can be found in supplementary table 1 in our publication: (link)."
#                     ], id="cohort-details"),
#                     html.Div(className="option-details", children=["Detailed sample preparation and LC-MS information can be found in our publication here (link). The data displayed in this database was acquired from the analysis of patient-matched tumor tissues and normal mucosa using a UPLC-ESI-QTOFMS (H-Class ACQUITY and Xevo G2-XS; Waters Corporation, Milford, MA, USA) was used for MS data acquisitionby RPLC ESI positive and HILIC ESI negative mode. We chose to make our data available in the format of this database, other data requests, along with protocols and codes can be made by email, please see contact us section."], id="preparation-details"),
#                     html.Div(className="option-details", children=["In this database we have displayed all metabolite features generated from the analysis of the tumor tissues and normal mucosa tissues, by electrospray ionization (ESI) mode; negative or positive. These features are displayed in Section 1. For subsequent sections we only display annotated metabolites. The level of annotation is defined by the metabolomics standards initiative (MSI) levels; Level 1:…….Level 2:…….Level 3:……..Metabolite identification methods are published in Jain. Et al…..(paper under submission)."], id="feature-details"),
#                     html.Div(className="option-details", children=[
#                              "Please cite the following: Jain A, ...paper details here"], id="link-details"),
#                     html.Div(className="option-details", children=[
#                              "The data acquired in this database was supported by funding from the American Cancer Society awarded to Caroline Johnson, and the Yale Center for Clinical Investigation awarded to Abhishek Jain."], id="project-details"),
#                     html.Div(className="option-details", children=[
#                              "Please contact Caroline Johnson: caroline.johnson@yale.edu  or Abhishek Jain: a.jain@yale.edu for any inquiries"], id="contact-details"),
#                 ])
#             ])
#         ],),
#     ]),
# ])
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
                    html.Div(className="tab", children=[
                        html.A("More Information",
                               className = "moreinfo",
                               href="#footer",)
                        ]
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
                        html.H1("Microbiome Metabolome Database",
                                className="title"),
                        html.P("About", className="about-text"),
                        html.P(
                            [
                                "This database is a comprehensive resource for exploring the metabolic activities of individual gut bacterial species and their roles in human health and disease. ",
                                html.Br(), 
                                html.Br(),
                                "It integrates data from 310 in vitro bacterial cultures and 112 monocolonized mouse models, each hosting a single bacterial species, providing insights into species-specific metabolic profiles in both controlled environments and host systems. ",
                                html.Br(), 
                                html.Br(),
                                "By comparing in vitro and in vivo metabolomics data, the database highlights key biochemical pathways and their significance in host-microbe interactions, including immune modulation and other physiological processes. ",
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
        dbc.Row(
            [
                html.H2(
                    "Different Heading",
                    className="section-heading",
                    id="section1",
                ),
                html.P(
                    "Something to write here.",
                    className="section-description",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                        [
                            html.Label(
                                        "Select Compound:",
                                        id="gmm-section",
                                        className="select-label",
                                    ), 
                            dcc.Dropdown(
                                    id="selected-compound-gmm",
                                    options=[
                                        {"label": name, "value": name}
                                        for name in list(get_gmm_name("gmm_test_1"))
                                    ],
                                    placeholder="Select Mz Value",
                                    searchable=True,
                                    clearable=True,
                                    multi=False,
                                    style={"width": "100%"},
                                    className="select-input",
                                    value=list(get_gmm_name("gmm_test_1"))[0],
                                ),
                            dcc.RadioItems(
                                id="data-display-selector",
                                options=[
                                    {"label": "Show All Data", "value": "all"},
                                    {"label": "Top 10 Values", "value": "top"},
                                    {"label": "Bottom 10 Values", "value": "bottom"},
                                ],
                                value="all",
                                labelStyle={"display": "inline-block"},
                                className="select-input",
                            ),
                            dcc.Loading(
                                id="outer-container-plus-loading",
                                type="circle",
                                children=[
                                    html.Div(
                                        [
                                            html.Div(
                                                id="selected-gmm-value",
                                                className="select-label",
                                            ),
                                        dcc.Graph(
                                            id='gmm-scatter-plot',
                                            className="gmm-scatter-plot",
                                        )
    
                            ],
                            className="outer-container",
                        ),
                    ],
                ),
            ],
            md=12,
        ),
        html.Div(className="border-line"),
                        
            dbc.Row([
                dbc.Col(
                    [
                        html.Label("Select Metabolites:", className="select-label"),
                        dcc.Dropdown(
                            id="selected-metabolites",
                            options=[
                                {"label": name, "value": name} for name in list(get_column_names("gmm_test_1"))
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
                dbc.Col(
                    [
                        html.Label("Select Bacteria:", className="select-label"),
                        dcc.Dropdown(
                            id="selected-bacteria",
                            options=[
                                {"label": name, "value": name} for name in list(get_bacteria_names("gmm_test_1"))
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
            ]),

            dcc.Loading(
                id="outer-container-plus-loading-heatmap",
                type="circle",
                children=[
                    html.Div(
                        [
                            dcc.Graph(
                                id='gmm-heatmap-plot',
                                className="gmm-heatmap-plot",
                            )
                        ],
                        className="outer-container",
                    ),
                ],
            ),


                    ]
                )

            ]
        ),
        html.Div(className="border-line"),
        # add tabs to it
        dbc.Row(
            [
                html.H2(
                    "Tumor vs. Normal Mucosa Metabolite Comparison Across Subsites",
                    className="section-heading",
                    id="section2",
                ),
                html.P(
                    "In this section, we present the comparison of identified metabolites between patient-matched tumor tissues and normal mucosa tissues in each of the seven subsites of the colorectum. This analysis focuses on 409 annotated metabolites; 220 were annotated using authentic standards by LC-MS (MSI levels 1 and 2), and 190 annotated were annotated using in silico methods (MSI level 3). Statistical significance was assessed using a paired Mann-Whitney U test, with all p-values adjusted for multiple comparisons using the Benjamini-Hochberg (BH) false discovery rate (FDR) method. Additionally, users can analyze the unique metabolite alteration specific to each subsite.",
                    className="section-description",
                ),
            ]),
        dbc.Row([
                dbc.Col([
                    html.Label(
                        "Select Filter:",
                        id="filter-section-meta",
                        className="select-label",
                    ),
                    dcc.RadioItems(
                        id="filter-radio",
                        options=[
                            {"label": "All metabolites", "value": "all"},
                            {"label": "Metabolites altered across all subsites",
                             "value": "across_all"},
                            {"label": "Subsites specific alterations",
                             "value": "specific_subsites"},
                            {"label": "Proximal or Distal subsites",
                             "value": "proximal_distal"},
                        ],
                        value="all",
                        inline=True,  # Display radio items horizontally
                        className="select-input",
                    ),
                ],
                    md=4,
                ),
                ]),
        dbc.Row([
                dbc.Col([
                    html.Label(
                        "Select Compound:",
                        id="meta-section",
                        className="select-label",
                    ),
                    dcc.Dropdown(
                        id="compound-dropdown-meta",
                        placeholder="Select Meta Value",
                        searchable=True,
                        multi=False,
                        style={"width": "100%"},
                        className="select-input",
                    ),
                    html.Div(
                        id="selected-meta-value",
                        className="select-label",
                    ),
                ],
                    md=12,
                ),
                ]),

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
                                                            id=f"scatter-plot-meta-{i}",
                                                            className="scatter-plot",
                                                        )
                                                        for i in range(7)
                                                    ],
                                                    className="inner-container",
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
        html.Div(className="border-line"),

        dbc.Row(
            [
                html.H2(
                    "Inter-subsite metabolites comparisons", 
                    className="section-heading",
                    id="section3",
                ),
                html.P(
                    "This section facilitates a comparative analysis of metabolite profiles within tumors and normal tissues separately, enabling users to explore inter-subsite metabolite variations across the seven CRC subsites. Moreover, users can discern differences in metabolite abundances among right-sided colorectal cancer, left-sided colorectal cancer, and rectal cancer, encompassing both tumor-tumor and normal-normal comparisons. Statistical significance was determined using Kruskal-Wallis ANOVA with Dunn’s post hoc test, with a two-sided adjusted p-value ≤ 0.05 indicating statistical significance.",
                    className="section-description",
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col([tabs_compare, html.Div(id="tabs-content")], md=12),
                # dbc.Col([dcc.Dropdown(id="image-dropdown", className="select-input",
                #                       options=get_dropdown_options(),
                #                       value=get_dropdown_options()[0]["value"])], md=4),
                # dbc.Col([
                #     html.Div([
                #         html.Img(id="selected-image", style={"width": "80%"}),
                #     ], id="selected-image-container")
                # ], md=8),
            ]
        ),

        html.Div(className="border-line"),
        dbc.Row(
            [
                html.H2(
                    "Linear metabolite gradient across colorectal subsites",
                    className="section-heading",
                    id="section4",
                ),
                html.P(
                    "This section enables users to analyze metabolite concentration gradients across the colorectum. A linear regression analysis was conducted between subsites and metabolite abundance to assess the presence of a consistent linear trend from the cecum to the rectum. A p-value ≤ 0.05 indicates a statistically significant linear relationship of the CRC subsites with metabolite abundance, suggesting a linear change in metabolite concentrations along the colorectal tract.",
                    className="section-description",
                ),
                dbc.Col(
                    [
                        html.Label(
                            "Select Compound:",
                            id="mz-linear-section",
                            className="select-label",
                        ),
                        dcc.Dropdown(
                            id="compound-dropdown-linear",
                            options=[
                                {"label": mz, "value": mz}
                                for mz in list(get_linear_values(["tumor_linear_plots", "normal_linear_plots"]))
                            ],
                            placeholder="Select Mz Value",
                            searchable=True,
                            clearable=True,
                            multi=False,
                            style={"width": "100%"},
                            className="select-input",
                            value=list(get_linear_values(
                                ["tumor_linear_plots", "normal_linear_plots"]))[0],
                        ),
                        html.Div(id="selected-mz-linear-value"),

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
                                        dcc.Graph(
                                            id="tumor-linear-plot",
                                            className="tumor-linear-plot",
                                        ),
                                        dcc.Graph(
                                            id="normal-linear-plot",
                                            className="normal-linear-plot",
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
        html.Div(className="border-line"),
        dbc.Row(
            [
                html.H2(
                    "Survival Metabolite Marker Comparison Across Colorectal Subsites",
                    className="section-heading",
                    id="section5",
                ),
                html.P(
                    "In this section, we present the comparison of survival metabolite markers between colorectal subsites. Cox proportional hazard regression analysis was performed to identify the association between log2 abundances of individual metabolites and 5-year overall survival in each subsite, adjusting for age, sex chemotherapy, and stage. A two-sided p-value less than 0.05 was considered statistically significant. Additionally, users can analyze the unique survival markers specific to each subsite",
                    className="section-description",
                ),

            ]),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Col([tabs_survival, html.Div(
                            id="tabs-content")], md=12),

                    ],
                    md=12,
                ),
            ]
        ),
        html.Div(className="border-line"),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Loading(
                            id="loading",
                            className="loading",
                            type="circle",
                            children=[html.Div(id="loading-output")],
                        ),
                    ],
                    md=12,
                ),
            ]
        ),
    footer_layout     
    ],
    fluid=True,
)
