import pandas as pd
import numpy as np
import sweetviz as sv
from faker import Faker

np.random.seed(0)

fake = Faker()
date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
df = pd.DataFrame({
    'date': pd.to_datetime(np.random.choice(date_range, 10000)),
    'price': np.random.uniform(10, 100, 10000),
    'ID': np.random.choice(['A', 'B', 'C'], 10000)
    })

report  = sv.analyze(df)
report.show_html("ego.html")
