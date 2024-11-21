import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sample data for Waterfall plot
x = ["A", "B", "C", "D"]
y = [10, -5, 15, -10]

# Define row heights: Increase heights for rows 5 to 8
row_heights = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2]

# Create the 8x4 subplot grid with customized row heights
fig = make_subplots(
    rows=8, cols=4,
    specs=[
        [{"rowspan": 3, "colspan": 4}, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
        [{}, {}, {}, {}],
        [{}, {}, {}, {}],
        [{}, {}, {}, {}],
        [{}, {}, {}, {}],
        [{}, {}, {}, {}],
    ],
    row_heights=row_heights,  # Set custom row heights
    print_grid=True
)

# Add a Waterfall plot to the specified position
fig.add_trace(go.Waterfall(x=x, y=y), row=1, col=1)
fig.update_xaxes()
fig.update_trace()
fig.update_layout( template='plotly_dark', plot_bgcolor='rgb(80,80,80)', #template='ggplot2'
                  title='title', title_x=0.5,
                  height=1500, width=1500
                 )

# Show the plot
fig.show()
