import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a 4x4 subplot grid
fig = make_subplots(rows=4, cols=4)

# Add 16 traces, specifying the row and column indirectly
for i in range(16):
    row = (i // 4) + 1  # Calculate the row number
    col = (i % 4) + 1   # Calculate the column number
    fig.add_trace(
        go.Scatter(x=[1, 2, 3], y=[i, i * 2, i * 3], mode='lines', name=f'Trace {i+1}'),
        row=row, col=col
    )

# Update layout
fig.update_layout(height=800, width=800, title_text="4x4 Subplots Example with Indirect Row and Column")
fig.show()