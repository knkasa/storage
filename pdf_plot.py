import pandas as pd
import numpy as numpy 
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import table
import japanize_matplotlib
import pdb

os.chdir("C:/Users/knkas/Desktop/NLP_example")

# Create a Pandas DataFrame with sample data
data = {
        'x1': [1, 2, 3, 4, 5],
        'y1': [10, 15, 13, 18, 20],
        'x2': [1, 2, 3, 4, 5],
        'y2': [5, 8, 9, 6, 12]
        }
df = pd.DataFrame(data)

pdf_pages = PdfPages('two_graphs.pdf')

# Add a Title page.
title_fig, title_ax = plt.subplots( figsize=(9,9) )
title_ax.text(0.5, 0.8, 'Title Page Text', ha='center', va='center', fontsize=16)  # center, top, bottom, left/right
title_ax.axis('off')  # Turn off axes for the title page
comments = [
    'Comment 1: This is the first comment.',
    'Comment 2: Another comment goes here.',
    'Comment 3: Add more comments as needed.'
    ]
for i, comment in enumerate(comments):
    title_ax.text(0.5, 0.8 - 0.1*(i + 1), comment, ha='center', va='center', fontsize=10)
pdf_pages.savefig(title_fig, bbox_inches='tight')

# Add the first figure page.
plt.figure()
plt.plot(df['x1'], df['y1'], label='Graph 1')
pdf_pages.savefig()

# Add the second figure page.
plt.figure()
plt.plot(df['x2'], df['y2'], label='Graph 2')
pdf_pages.savefig()

# Create a figure for the DataFrame table and add it to the PDF on the third page
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 2))
ax[0].axis('off')  # Hide axes
tbl = table(ax[0], df, loc='center', cellLoc='center', colWidths=[0.2]*len(df.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
ax[1].axis('off')  # Hide axes
tbl = table(ax[1], df, loc='center', cellLoc='center', colWidths=[0.2]*len(df.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
pdf_pages.savefig()

pdf_pages.close()


#Alternative
'''
fig1, ax1 = plt.subplots()
ax1.plot(x, y1)
ax1.set_title('Figure 1')

fig2, ax2 = plt.subplots()
ax2.plot(x, y2)
ax2.set_title('Figure 2')

pdf_file = 'multipage_plot.pdf'
with PdfPages(pdf_file) as pdf:
    pdf.savefig(fig1)
    pdf.savefig(fig2)
'''


