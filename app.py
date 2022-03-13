from time import time
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import main

Scatter = main.Scatter()
scatter_df = Scatter.scatter_df
scatter_df.loc[scatter_df['Predicted goal pace']<0,'Predicted goal pace']=0
scatter_df = scatter_df.round(decimals=1)


app = Dash(__name__)
server = app.server

font = "Arial"
colors = {
    'background': '#0E1012',
    'text': '#A0AABA'
}

color_dict = {
    "ANA": "#F47A38",
    "ARI": "#8C2633",
    "BOS": "#FFB81C",
    "BUF": "#002654",
    "CGY": "#C8102E",
    "CAR": "#CC0000",
    "CHI": "#CF0A2C",
    "COL": "#6F263D",
    "CBJ": "#002654",
    "DAL": "#006847",
    "DET": "#CE1126",
    "EDM": "#FF4C00",
    "FLA": "#B9975B",
    "L.A": "#A2AAAD",
    "LAK": "#A2AAAD",
    "MIN": "#154734",
    "MTL": "#AF1E2D",
    "NSH": "#FFB81C",
    "N.J": "#c8102e",
    "NJD": "#c8102e",
    "NYI": "#00539B",
    "NYR": "#0038A8",
    "OTT": "#C52032",
    "PHI": "#F74902",
    "PIT": "#FCB514",
    "SEA": "#99D9D9",
    "STL": "#002F87",
    "S.J": "#006D75",
    "SJS": "#006D75",
    "T.B": "#002868",
    "TBL": "#002868",
    "TOR": "#00205B",
    "VAN": "#00843D",
    "VGK": "#B4975A",
    "WSH": "#C8102E",
    "WPG": "#041E42"
    }

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        style={
            'textAlign': 'center',
            'color': colors['text'],
            "font-family": font
        }, 
        children='Outlier Detection in NHL Goal Scoring'
        ),

    html.Div(
        style={
            'textAlign': 'center',
            'color': colors['text'],
            "font-family": font
        }, 
        children='''
        An ensemble model for finding under/overperformers.
    '''),

    html.Div(
        style={
            'textAlign': 'center',
            'color': colors['text'],
            "font-family": font
        }, 
        children='''
        Players above the red line scored more goals than the model predict; players who are below the line scored less than predicted.
    '''),

    html.Div(
        style={
            'textAlign': 'center',
            'color': colors['text'],
            "font-family": font
        }, 
        children='''
        Try clicking on individual players to open up their scoring history compared to the model's predictions.
    '''),

    html.Div(
        style={
            'textAlign': 'center',
            'color': colors['text'],
            "font-family": font
        }, 
        children='''
        Double click on teams to isolate them. Zoom by clicking and dragging.
    '''),

    html.Div(html.P([html.Br()])),

    dcc.Slider(
        scatter_df['Season'].min(),
        scatter_df['Season'].max(),
        step=None,
        value=2020,
        marks={str(year): str(year) for year in scatter_df['Season'].unique()},
        id='year-slider'
    ),

    html.Div(
        [
            dcc.Graph(
                id='graph-with-slider', 
                clickData={'points': [{'hovertext': 'Jakob Chychrun'}]}
                ),
        ], 
        style={'display': 'inline-block', 'width': '60%'}
        ),
    
    html.Div([
        dcc.Graph(id='time-series')
    ], style={'display': 'inline-block', 'width': '39%'})

])

@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):

    selected_df = scatter_df[scatter_df["Season"] == selected_year]

    fig = go.Figure()
    fig = px.scatter(selected_df, x="Predicted goal pace", y="Goal pace",
                    size="Goal pace", hover_name="Player", color="Team",
                    color_discrete_map = color_dict,
                    log_x=False, size_max=15)

    real_goal_max = selected_df["Goal pace"].max()
    pred_goal_max = selected_df["Predicted goal pace"].max()
    goal_max = np.max([real_goal_max, pred_goal_max])

    fig.add_trace(go.Scatter(x=[0.01,goal_max],y=[0.01,goal_max],line_color='#FF0000'))

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        transition_duration=500
    )

    return fig

@app.callback(
    Output('time-series', 'figure'),
    Input('graph-with-slider', 'clickData'))
def update_time_series(clickData):

    player = clickData["points"][0]["hovertext"]

    time_series_df = scatter_df[scatter_df["Player"] == player]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_series_df["Season"].tolist(),
        y=time_series_df["Goal pace"].tolist(),
        name="Actual"       # this sets its legend entry
    ))


    fig.add_trace(go.Scatter(
        x=time_series_df["Season"].tolist(),
        y=time_series_df["Predicted goal pace"].tolist(),
        name="Predicted"
    ))

    fig.add_annotation(x=0.05, y=0.95, xanchor='left', yanchor='bottom',
                        xref='paper', yref='paper', showarrow=False, align='left',
                        text='<b>{}</b><br>{}'.format("Player:", player))

    fig.update_layout(
        xaxis_title="Season",
        yaxis_title="Goal Pace",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        transition_duration=500
    )








    # fig = px.scatter(time_series_df, x='Season', y=['Goal pace',"Predicted goal pace"])

    # fig.update_traces(mode='lines+markers')

    # fig.add_annotation(x=0.05, y=0.95, xanchor='left', yanchor='bottom',
    #                    xref='paper', yref='paper', showarrow=False, align='left',
    #                    text='<b>{}</b><br>{}'.format("Player:", player))

    # fig.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text'],
    #     transition_duration=500
    # )
    # fig.update_layout(showlegend=True)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)