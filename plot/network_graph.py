import networkx as nx
import matplotlib.pyplot as plt

#====== Create a graph with weighted edge =========
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

# Use the previous graph W
shortest_path = nx.shortest_path(W, source='A', target='C', weight='weight')
print(f"Shortest path from A to C: {shortest_path}")

degree_dict = dict(W.degree())
print(f"Node degrees: {degree_dict}")

clustering = nx.clustering(W)
print(f"Clustering coefficient: {clustering}")

#===== Another example with no weights ===========
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()

# Add nodes
G.add_node(1)
G.add_nodes_from([2, 3, 4])  # Adding multiple nodes at once

# Add edges
G.add_edge(1, 2)
G.add_edges_from([(2, 3), (3, 4), (4, 1)])  # Adding multiple edges

# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', node_size=1500, font_size=15)
plt.title("Graph Visualization")
plt.show()


