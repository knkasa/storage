import matplotlib.pyplot as plt

fig, ax = plt.subplot()
plt.plot( x, y, label='xxxx')
ax.legend(loc='upper left')
ax.grid()
ax2 = ax.twinx()
plt.plot( x, y2, label='xxxx2')
ax2.legend(loc='upper right'):
ax2.grid()
plt.show()
