import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()
G.add_edge("A", "B")
G.add_edge("B", "C")
G.add_edge("C", "A")
G.add_edge("C", "D", weight=0.7)

# Analyze the graph
print("Nodes:", G.nodes)
print("Edges:", G.edges)
print("Degree of each node:", dict(G.degree))

# Draw the graph
nx.draw(G, with_labels=True, node_color="lightblue", font_weight="bold")
plt.show()
