import pandas as pd
import numpy as numpy 
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import table
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

plt.figure()
plt.plot(df['x1'], df['y1'], label='Graph 1')
pdf_pages.savefig()

plt.figure()
plt.plot(df['x2'], df['y2'], label='Graph 2')
pdf_pages.savefig()

# Create a figure for the DataFrame table and add it to the PDF on the third page
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')  # Hide axes
tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.2]*len(df.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
pdf_pages.savefig()

pdf_pages.close()

