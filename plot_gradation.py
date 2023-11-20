import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from matplotlib.font_manager import FontProperties
import japanize_matplotlib
import pandas as pd
import numpy as np

# kinds of colors.  https://matplotlib.org/stable/users/explain/colors/colormaps.html

def hex_to_RGB(hex_str):
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

df_bar = pd.DataFrame({ 'あい':[10], 'うめ':[15], 'そら':[20] }).T

#japanese_font = FontProperties(fname='path/to/japanese_font.ttf') 
sns.set_theme(style='darkgrid')
sns.set(font='IPAexGothic')  # enable Japanese font in seaborn.

figure(num=None, figsize=(7, 4), dpi=100, facecolor='w', edgecolor='k') 
plt.barh( df_bar.index, df_bar[0].values, color=get_color_gradient("#8A5AC2", "#3575D5", len(df_bar)) )
plt.show()








