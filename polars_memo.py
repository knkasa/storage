import pandas as pd
import pdb
import polars as pl
import numpy as np

num_records = 1_000_000

types = np.random.choice(['A', 'B', 'C'], size=num_records)
values = np.random.uniform(low=0.0, high=1000.0, size=num_records)

df = pd.DataFrame({'Type': types,'Val': values})
df = pl.from_pandas(df)

df.head()
df.tail()
df.schema
df.columns
df.describe()

df[0,0]
df['Val']

df.fill_nan()
df.fill_null()

df.rename({'old':'new'})
df.clone() # deepcopy

# change type.
df = d.with_column(pl.col('Val').cast(pl.Float64))

# Apply method.
def fun(x):
  return df['Val']*2
df.with_column(pl.struct(["Type","Val"]).map_elements(fun,return_dtype=pl.Float64).alias("one") )

# convert datetime
df = df.with_column( pl.col('date').str.strptime(pl.Date, '%Y-%m-%d').alias('date') )

# concat 
df = df.concat( [df1, df2], how='vertical' ) 

# Groupby
result = df.group_by("col").agg([ pl.col("Val").mean().alias("mean_val"), ... ])

# Groupby with custom function
def fun2(x):
  return x[0].mean()  # [0] corresponds to 'Val' column.
df.group_by('Type').agg( pl.map_groups(exprs=['Val'], function=lambda x: fun2(x) ).alias('result') )

# Select command.
df.filter( (pl.col('Type').is_in(['A', 'B'])) & (pl.col('Val') > 0) )

# update
df = df.with_column(pl.when(pl.col('Type') == 'A').then(pl.lit(100.0)).otherwise(pl.col('Val')).alias('Val') )

# deletion.
df = df.drop(['B', 'C'])
df.unique( subset=['Val'], keep='first' ) 
df.is_duplicate() # this is faster.

# Join
merged_df = df1.join(df2, on='ID', how='inner')


