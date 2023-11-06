import pandas as pd
import numpy as np
from faker import Faker
import missingno as msno
#import pdb
import matplotlib.pyplot as plt

# If error "soft unicode from Markupsafe" appears, run "python -m pip install markupsafe==2.0.1"

np.random.seed(0)

fake = Faker()
date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
df = pd.DataFrame({
    'date': pd.to_datetime(np.random.choice(date_range, 10000)),
    'price': np.random.uniform(10, 100, 10000),
    'ID': np.random.choice(['A', 'B', 'C'], 10000)
    })

#pdb.set_trace()
msno.bar(df)
msno.heatmap(df)
#msno.matrix(df)
plt.show()
