import matplotlib.pyplot as plt
import numpy as np

# color kinds.  https://matplotlib.org/stable/users/explain/colors/colormaps.html

labels = ['Category A', 'Category B', 'Category C', 'Category D']
sizes = [30, 25, 20, 25]

# 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
# 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
# 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'

# 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
# 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
# 'tab20c'

cmap = plt.get_cmap('Pastel1')
colors = cmap(np.linspace(0, 1.0, len(labels))) 

plt.pie( sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors )
plt.axis('equal')
plt.title('Pie Chart Example')
plt.show()