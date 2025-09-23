import plotly.graph_objects as go
import pandas as pd

data = {
    'source': ['Supplier A', 'Supplier A', 'Supplier B', 'Supplier B', 
               'Factory X', 'Factory X', 'Factory Y', 'Factory Y',
               'Warehouse 1', 'Warehouse 2'],
    'target': ['Factory X', 'Factory Y', 'Factory X', 'Factory Y',
               'Warehouse 1', 'Warehouse 2', 'Warehouse 1', 'Warehouse 2',
               'Retailer 1', 'Retailer 2'],
            # (sourcr->target=amout) supplierA->FactoryX=500, SupplierA->FactoryY=300, ... Warehouse2->Retailer2=600
    'value': [500, 300, 400, 600, 600, 300, 400, 500, 700, 600],
    'color': ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4',
              '#45B7D1', '#45B7D1', '#96CEB4', '#96CEB4',
              '#FECA57', '#FF9FF3']
    }

df = pd.DataFrame(data)

all_nodes = list(set(df['source'].unique().tolist() + df['target'].unique().tolist()))
node_indices = {node: i for i, node in enumerate(all_nodes)}

fig = go.Figure(go.Sankey(
    node=dict(
        pad=20,
        thickness=25,
        line=dict(color="black", width=1),
        label=all_nodes,
        color=[data['color'][i] for i in range(len(all_nodes))]
    ),
    link=dict(
        source=[node_indices[src] for src in df['source']],
        target=[node_indices[tar] for tar in df['target']],
        value=df['value'],
        color=[f"rgba{tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) + (0.4,)}" 
               for color in df['color']]
    )
    ))

fig.update_layout(
    title_text="Supply Chain Flow Sankey Diagram",
    font_size=14,width=1000,height=600,
    paper_bgcolor='lightgray'
    )
fig.show()