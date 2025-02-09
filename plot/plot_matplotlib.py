# If you need to show japanese characters, use
#https://albertauyeung.github.io/2020/03/15/matplotlib-cjk-fonts.html/

import japanize_matplotlib  # 日本語表記のmaplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')  # need to import "figure" module. 
plt.plot( x,y , 'bo', label='line1'  )   # need to put labels for legend 
plt.plot( x2, y2, 'k-', label='line2' )
plt.legend( loc=('upper right'), bbox_to_anchor=(1.2, 0.5) ); # use either loc or bbox to set legend position.
plt.xlabel('x'); plt.ylabel('y') 
plt.xticks(rotation=45);  plt.xticks([]);   #remove x-axis  
plt.xticks(np.arange(0, len(x)+1, 5))  ;   # show only xlabel every 5 points
plt.tight_layout()  # fit graph size to window
plt.xlim(x_min, x_max)  # define axis range  

plt.show() 
plt.savefig('../image.png' ,  dpi=400 )
plt.clf()   # this clears plot.  (useful when using loop)
