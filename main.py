import os
os.environ['dash_app_root'] = os.path.dirname(os.path.abspath(__file__))
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'

from dotenv import load_dotenv
load_dotenv()

# Import required libraries
import datetime
import time


import pandas as pd
import dash
from dash import html
from plotly import express as px
from dash import dcc, clientside_callback, Patch
from dash.dependencies import Input, Output, State

##========================================
####=======================================
######======================================
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from utilities import isDebug

base = 'https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.css'
theme = str(dbc.themes.CERULEAN)

load_figure_template(['cerulean', 'cerulean_dark'])
template = 'cerulean'
map_styles = ['open-street-map', 'carto-darkmatter']
states = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
app = dash.Dash(
    __name__,
    external_stylesheets=[base, theme, dbc.icons.FONT_AWESOME, dbc_css],
    title="WEeDetect App",
    use_pages=True,
    suppress_callback_exceptions=True,
)

# from plotly.io import templates
# print(list(templates))
### Alternative is to use Dash Leaflet - But on App Engine additional packages may limit Free Tier usage.
australian_center = {"lat": -22.867313536366957, "lon": 133.9434557416898}
not_display_nav = '' if len(dash.page_registry.values()) > 1 else ' d-none'

app.layout = dbc.Container([
        dcc.Store(id='theme', data="plotly_white"),
        dcc.Location(id='url', refresh=False),
        dbc.Row([
            dbc.Col([
                    dbc.DropdownMenu(
                        label=html.I(className='fa-solid fa-bars', style={"fontSize":"15px", "cursor": "pointer"}),
                        children=[
                            dbc.DropdownMenuItem(
                                page['name'], id=page['name'], href=page["path"], class_name='dropdown-item-sm rounded-0'
                            ) for page in dash.page_registry.values()
                        ],
                        class_name='dropdown-menu-sm d-lg-none rounded-0 bg-body'+ not_display_nav,
                        size='sm',
                        # color='dark',
                        id='nav-menu',
                    ),
                ] ,xs=2, lg=3),
            dbc.Col(html.H2("WEeDetect App",id="header-title", className='display-4 text-center pt-2 pb-2'), xs=10, md=6),
            dbc.Col(html.Div([
                    html.A(href="https://github.com/RayGone/Dash", target="_blank", title='Check Github', className='fa-brands fa-github me-3', style={"fontSize":"20px", "cursor": "pointer"}),
                    dbc.Label(class_name="fa fa-moon pe-2", html_for="switch", style={"color":"silver", "textShadow":"1px 0 2px black"}),
                    dbc.Switch(id="switch", value=True, class_name="d-inline-block", persistence=True),
                    dbc.Label(class_name="fa fa-sun", html_for="switch", style={"color":"#EABF14", "textShadow":"1px 1px 2px #aaa, 0 0 2px #EABF14"})
                ], className='d-inline-block float-end d-print-none', style={"whiteSpace":"nowrap"}), align='center', md=3)
        ],align='center', justify='between', key='row1', class_name='sticky-top shadow-sm mb-3 bg-body'),
        
        html.Div([
            dbc.Nav(
                [
                    dbc.NavItem(
                        dbc.NavLink(
                            html.Div(page["name"]),
                            href=page["path"],
                            active="exact",
                            class_name='rounded-0 nav-link-sm',
                        ), class_name='nav-item-sm border-left rounded-left')
                    for page in dash.page_registry.values()
                ],
                vertical=True,
                pills=True,
                style={"top": "100px", "minWidth": "200px", "height": "calc(100vh - 120px)"},
                class_name="d-none sticky-bottom pt-1 start-0 border-0 rounded-top bg-body d-print-none bg-gradient" + ' d-lg-block' if not not_display_nav else not_display_nav,
                id="sidebar",
            ),
            html.Div(dash.page_container, className="flex-grow-1 flex-shrinkable ms-2")
        ], className='d-flex flex-column flex-lg-row g-1'),
    ], fluid=True, class_name='dbc')

@app.callback(Output("header-title", "children"),   Input("url", "pathname"))
def update_header_title(pathname):
    for page in dash.page_registry.values():
        if pathname == page["path"]:
            return page["name"]
    return ""

@app.callback(Output("theme","data"), Input("switch", "value"))
def themeMode(mode):
    return template if mode else template+'_dark'

@app.callback(Output("nav-menu", "color"),
              Input("theme", "data"))
def update_nav_menu_color(theme):
    if 'dark' in theme:
        return 'dark'
    else:
        return 'light'

clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');
       return window.dash_clientside.no_update
    }
    """,
    Output("switch", "id"),
    Input("switch", "value"),
)

server = app.server
if __name__ == "__main__":
    app.run(debug=isDebug(), host="0.0.0.0", port="8088")