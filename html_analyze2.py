import pandas as pd
import numpy as np
from faker import Faker
from ydata_profiling import ProfileReport
import os
import pdb

# For displaying Japanese texts, see here.  https://zenn.dev/misaya/articles/e6c815da5056f5

os.chdir("C:/Users/knkas/Desktop/NLP_example")

# https://github.com/ydataai/ydata-profiling
# pip install ydata-profiling

np.random.seed(0)

fake = Faker()
date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
df = pd.DataFrame({
    'date': pd.to_datetime(np.random.choice(date_range, 10000)),
    'price': np.random.uniform(10, 100, 10000),
    'ID': np.random.choice(['A', 'B', 'C'], 10000)
    })

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")
pdb.set_trace()

