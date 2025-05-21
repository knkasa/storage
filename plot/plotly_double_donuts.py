import pandas as pd
import plotly.graph_objects as go

# Sample data
df = pd.DataFrame({
    'colA': ['a', 'b', 'b', 'a', 'a', 'b', 'a', 'b', 'b', 'a'],
    'colB': [1, 1, 0, 1, 0, 0, 1, 1, 0, 0]
})

# Count occurrences for each (colB, colA)
counts = df.groupby(['colB', 'colA']).size().reset_index(name='count')

# Separate data for inner (colB=1) and outer (colB=0)
inner = counts[counts['colB'] == 1]
outer = counts[counts['colB'] == 0]

# Ensure all colA values are present in both for consistent colors
all_colA = sorted(df['colA'].unique())
for val in all_colA:
    if val not in inner['colA'].values:
        inner = pd.concat([inner, pd.DataFrame({'colB': [1], 'colA': [val], 'count': [0]})])
    if val not in outer['colA'].values:
        outer = pd.concat([outer, pd.DataFrame({'colB': [0], 'colA': [val], 'count': [0]})])

# Sort to keep consistent color mapping
inner = inner.sort_values('colA')
outer = outer.sort_values('colA')

# Define consistent colors for colA values
color_map = {'a': 'royalblue', 'b': 'orange'}
colors = [color_map[val] for val in inner['colA']]

# Create the figure with two donut layers
fig = go.Figure()

fig.add_trace(go.Pie(
    labels=inner['colA'],
    values=inner['count'],
    name='colB = 1',
    hole=0.3,
    direction='clockwise',
    sort=False,
    marker=dict(colors=colors),
    textinfo='percent',
    domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]}, # controle the size and position
    scalegroup='donut',
    showlegend=False,
    textposition='inside', #auto
))

fig.add_trace(go.Pie(
    labels=outer['colA'],
    values=outer['count'],
    name='colB = 0',
    hole=0.7,
    direction='clockwise',
    sort=False,
    marker=dict(colors=colors),
    textinfo='percent+label',
    domain={'x': [0, 1], 'y': [0, 1]}, #control the size and position
))

fig.update_layout(
    title_text="Double Donut Chart: colA Distribution by colB",
    annotations=[dict(text='colB=1', x=0.5, y=0.5, font_size=16, showarrow=False)]
)

fig.show()
