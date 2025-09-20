
# Slope graph
import matplotlib.pyplot as plt
cats = ["A","B","C","D"]
t0 = [42, 55, 30, 60]
t1 = [50, 48, 39, 72]

for i,c in enumerate(cats):
    plt.plot([0,1],[t0[i], t1[i]], marker="o")
    plt.text(-0.05, t0[i], c, ha="right", va="center")
    plt.text(1.05, t1[i], f"{t1[i]}", ha="left", va="center")

plt.xticks([0,1], ["Q1","Q2"]); plt.yticks([])
plt.title("Customer Satisfaction: Q1 → Q2")
plt.tight_layout(); plt.show()
plt.close()

# Dumbbel Plot. 
import numpy as np
import pandas as pd

data = {
    'Category': ['A', 'B', 'C', 'D', 'E'],
    'Start': [10, 15, 8, 20, 12],
    'End': [25, 18, 22, 30, 28]
    }
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
for i, row in df.iterrows():
    plt.plot([row['Start'], row['End']], [i, i], 
             color='gray', alpha=0.7, linewidth=2)
plt.scatter(df['Start'], df.index, color='blue', s=100, label='Start', zorder=3)
plt.scatter(df['End'], df.index, color='red', s=100, label='End', zorder=3)
plt.yticks(df.index, df['Category'])
plt.xlabel('Values')
plt.title('Dumbbell Plot')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
plt.close()


# Bullet chart.   Bar chart with target KPI(vertical line)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
def create_bullet_chart(actual_value, target_value, ranges, range_colors, title):
    fig, ax = plt.subplots(figsize=(10, 2))
    max_range = max(ranges)
    for i in range(len(ranges) - 1):
        ax.add_patch(Rectangle((ranges[i], 0.1), 
                              ranges[i+1] - ranges[i], 
                              0.8, 
                              facecolor=range_colors[i],
                              alpha=0.3))
    
    ax.barh(0.5, actual_value, height=0.6, color='black', alpha=0.7, label='Actual')
    ax.axvline(x=target_value, ymin=0.1, ymax=0.9, color='red', linewidth=3, label='Target')
    ax.set_xlim(0, max_range * 1.1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Performance')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    plt.close()

ranges = [0, 50, 75, 90, 100]  # Poor, Fair, Good, Excellent
range_colors = ['#ff6b6b', '#ffd166', '#06d6a0', '#118ab2']
create_bullet_chart(85, 90, ranges, range_colors, 'Sales Performance (%)')


# Calendar heat map
#import calmap