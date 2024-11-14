import plotly.graph_objects as go

# Example data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 15, 13, 17, 22, 18, 16, 19, 25, 23]

# Create the figure
fig = go.Figure()

# Add the line trace
fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

# Update x-axis to show labels every 2 data points
fig.update_xaxes(dtick=2)

# Show the figure
fig.show()
