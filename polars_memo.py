import pandas as pd
import pdb
import polars as pl
import numpy as np

num_records = 1_000_000

types = np.random.choice(['A', 'B', 'C'], size=num_records)
values = np.random.uniform(low=0.0, high=1000.0, size=num_records)

df = pl.read_csv('xxxx.csv', columns=['col1', 'col2'] ) 

df = pd.DataFrame({'Type': types,'Val': values})
df = pl.from_pandas(df)

df.head()
df.tail()
df.schema
df.columns
df.describe()

df[0,0]
df['Val'].to_numpy()

df.fill_nan(0)
df.fill_null(0)

# drop duplicates.
df.unique('col1')   

df.rename({'old':'new'})
df.clone() # deepcopy

# change type.
df = d.with_columns(pl.col('Val').cast(pl.Float64))

# sorting.
df.sort("Type", descending=True)

# Apply method.
def fun(x):
  return df['Val']*2
df.with_columns(pl.struct(["Type","Val"]).map_elements(fun,return_dtype=pl.Float64).alias("one") )
#df.with_columns(pl.col("Val").map_elements(fun, return_dtype=pl.Float64).alias("one") )
#df.with_columns(pl.col("Val").map_elements(lambda x: fun(x), return_dtype=pl.Float64).alias("one") ) This work too.
df.with_columns(pl.col("Val").map_elements(lambda x: x*2.0, return_dtype=pl.Float64).alias("one") )

# convert datetime
df = df.with_columns( pl.col('date').str.strptime(pl.Date, '%Y-%m-%d').alias('date') )
df = df.with_columns( pl.col('date').str.to_datetime('%Y-%m-%d').alias('date') )  # could use to_datetime('%Y-%m-%d %H:%M:%S')

# concat 
df = df.concat( [df1, df2], how='vertical' ) 

# Groupby
result = df.group_by("col").agg([ pl.col("Val").mean().alias("mean_val"), ... ])
df.group_by('col').agg( pl.count() ).alias('count_rows')

# Groupby with custom function
def fun2(x):
  return x[0].mean()  # [0] corresponds to 'Val' column.
df.group_by('Type').agg( pl.map_groups(exprs=['Val'], function=lambda x: fun2(x), return_dtype=pl.Float32 ).alias('result') )
#df.group_by('Type').agg( pl.map_groups(exprs=['Val'], function=fun2, return_dtype=pl.Float32 ).alias('result') )  This work too.
df.group_by('Type').agg( pl.struct(['Type','Val']).map_elements(lambda x: fun2(x), return_dtype=pl.Flot32 ).alias('result') )  This work too.
df.group_by('Type').agg( pl.col('Val').map_elements(lambda x: x.mean(), return_dtype=pl.Float32).alias('result')

# Select command.
df.filter( (pl.col('Type').is_in(['A', 'B'])) & (pl.col('Val') > 0) )
df.filter( (pl.col('datetime')>pl.datetime(2024,1,31))  )
df.select(['col1', 'col2'])

# update
df = df.with_column(pl.when(pl.col('Type') == 'A').then(pl.lit(100.0)).otherwise(pl.col('Val')).alias('Val') )
df.with_columns( pl.lit(20.0).alias('new') ) 
df.with_columns( pl.Series( npl.random.rand(num_rows) ).alias('new')
df = df.with_columns((pl.col("Val") * pl.col("Val")).alias("new"))

# deletion.
df = df.drop(['B', 'C'])
df.unique( subset=['Val'], keep='first' ) 
df.is_duplicate() # this is faster.

# Join
df2 = pl.DataFrame({ 'Type':['A','B'], 'class':['left','right'] })

merged_df = df.join(df2, on='ID', how='inner')


