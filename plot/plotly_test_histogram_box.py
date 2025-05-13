import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Sample data
np.random.seed(0)
df = pd.DataFrame({
    "target": np.concatenate([np.random.normal(0, 1, 500), np.random.normal(2, 1, 500)]),
    "input": ["A"] * 500 + ["B"] * 500
})
df_A = df[df["input"] == "A"]
df_B = df[df["input"] == "B"]

fig = go.Figure()

# Box plots (on secondary x-axis domain)
fig.add_trace(go.Box(
    x=df_A["target"],
    name="A",
    marker_color='blue',
    boxpoints='outliers',
    yaxis='y3'
))
fig.add_trace(go.Box(
    x=df_B["target"],
    name="B",
    marker_color='red',
    boxpoints='outliers',
    yaxis='y3'
))

# Histograms
fig.add_trace(go.Histogram(
    x=df_A["target"],
    name="A",
    marker_color='blue',
    yaxis='y1',
    opacity=0.6,
))
fig.add_trace(go.Histogram(
    x=df_B["target"],
    name="B",
    marker_color='red',
    yaxis='y2',
    opacity=0.6,
))

# Layout with secondary axes and shared x-axis
fig.update_layout(
    barmode='overlay',
    height=600,
    yaxis=dict(
        title='Count A',
        titlefont=dict(color='blue'),
        tickfont=dict(color='blue'),
        domain=[0.0, 0.6]
    ),
    yaxis2=dict(
        title='Count B',
        titlefont=dict(color='red'),
        tickfont=dict(color='red'),
        overlaying='y',
        side='right'
    ),
    yaxis3=dict(  # For box plot
        domain=[0.65, 1],
        showticklabels=False
    ),
    xaxis=dict(
        domain=[0.0, 1.0],
        anchor='y'
    ),
    title="Overlay Histogram with Dual Y-Axes and Box Plot",
    showlegend=True
)

fig.show()
