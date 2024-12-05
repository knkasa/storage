import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Sample Data
df = px.data.gapminder()  # Contains country, year, life expectancy, GDP, etc.

app = dash.Dash(__name__)

# App Layout
app.layout = html.Div([
    html.H1("Interactive Dashboard", style={'textAlign': 'center'}),
    
    # Dropdown for filtering
    html.Div([
        html.Label("Select Continent:"),
        dcc.Dropdown(
            id='continent-dropdown',
            options=[{'label': c, 'value': c} for c in df['continent'].unique()],
            value='Asia'  # Default value
            )
        ], style={'width': '50%', 'margin': 'auto'}),
    
    # Define ID for scatter plot.
    dcc.Graph(id='scatter-plot'),
    
    # Define ID for bar plot.
    dcc.Graph(id='bar-chart')
    ])

# Callbacks for Interactivity
@app.callback(
    [Output('scatter-plot', 'figure'), Output('bar-chart', 'figure')],
    [Input('continent-dropdown', 'value')]
    )
def update_charts(selected_continent):
    # Filter data
    filtered_df = df[df['continent'] == selected_continent]

    # Scatter Plot
    scatter_fig = px.scatter(
        filtered_df,
        x='gdpPercap',
        y='lifeExp',
        color='country',
        size='pop',
        hover_name='country',
        title=f"Life Expectancy vs GDP per Capita ({selected_continent})"
        )

    # Bar Chart
    bar_fig = px.bar(
        filtered_df.groupby('year')['pop'].sum().reset_index(),
        x='year',
        y='pop',
        title=f"Total Population over Time ({selected_continent})"
        )

    return scatter_fig, bar_fig

if __name__ == '__main__':
    app.run_server(debug=True)
