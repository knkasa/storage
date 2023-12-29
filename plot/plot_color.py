import matplotlib.pyplot as plt
import numpy as np

x = (0.1, 0.3, 0.5, 1.0)
z = [1, 2, 3, 4]  # Replace with your z values

cmap = plt.cm.viridis  # You can choose a different colormap
colors = cmap(np.linspace(0, 1, len(x)))

# Plot the data with the specified colors
plt.plot(x, z, color=colors)