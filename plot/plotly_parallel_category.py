import plotly.graph_objects as go
import pandas as pd

# Your data
data = {
    'Group A': {'green': 10, 'blue': 23, 'yellow': 15, 'red': 20},
    'Group B': {'green': 43, 'blue': 11, 'yellow': 14},
    'Group C': {'green': 32, 'blue': 43, 'yellow': 32, }
    }

# Prepare data for parallel categories plot
# We need to create individual records for each count
records = []
for group, colors in data.items():
    for color, count in colors.items():
        for _ in range(count):
            records.append({'Group': group, 'Color': color})

df = pd.DataFrame(records)

# Create the parallel categories plot
fig = go.Figure(data=[go.Parcats(
    dimensions=[
        {'label': 'Group',
         'values': df['Group']},
        {'label': 'Color',
         'values': df['Color']}
        ],
    # Color by the Color dimension
    line={'color': df['Color'].map({'green': 'green', 'blue': 'blue', 'yellow': 'gold', 'red':'orange'}),
          'colorscale': 'Viridis'},
    hoveron='color',
    hoverinfo='count+probability',
    arrangement='freeform'
    )])

fig.update_layout(
    title='Parallel Categories Plot: Groups and Colors',
    font_size=12,
    #width=800, height=500
)

fig.show()

