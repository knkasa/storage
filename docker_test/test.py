import os
import pandas as pd
import numpy as np

print("testing docker on local")
print(os.environ.get('MY_STRING'))
print(os.getenv("MY_ENV_VAR"))
df = pd.DataFrame({'col':np.random.rand(10)})
df.to_csv('/my_dir/res.csv')