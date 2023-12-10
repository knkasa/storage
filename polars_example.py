import numpy as np
import polars as pl
import pdb

# Usage of polars which is based on Rust.  

df = pl.DataFrame({
    'price': np.random.uniform(10, 100, 10000),
    'ID': np.random.choice(['A', 'B', 'C'], 10000)
    })
    
pdb.set_trace()

#df = pl.read_csv("https://j.mp/iriscsv")                         # データ読み込み

# Filter rows
df.filter(df['price']>23)

# Adding a new column
df = df.with_columns( (pl.col('price')*2).alias("double_price") )

# Groupby
df_group = df.groupby('ID').agg( pl.col('price').mean().alias('average_price') )

# join
df2 = pl.DataFrame({'ID':['A','B'], 'type':['left','right']})
df.join( df2, on='ID', how='left' )

'''
df_agg = (
    df
    .select([pl.col("^sepal_.*$"), pl.col("species")])           # 列の選択
    .with_columns((pl.col("sepal_width") * 2).alias("new_col"))  # 列の追加
    .filter(pl.col("sepal_length") > 5)                          # 行の選択
    .group_by("species")                                         # グループ化
    .agg(pl.all().mean())                                        # 全列に対して平均を集計
)
'''

