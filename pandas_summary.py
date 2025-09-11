import pandas as pd
import numpy as np
import datetime as dt

# PandasAI 
# https://medium.com/@fareedkhandev/pandas-ai-the-future-of-data-analysis-8f0be9b5ab6f

# save/read to csv
df = pd.read_csv('C:/Users/ken_nakatsukasa/Desktop/python_code/data.csv', encoding='shift-jis') # try "utf-8" if not work. 
df.to_csv('C:/Users/ken_nakatsukasa/Desktop/python_code/result.csv', index=None, usecols=['cos1','col2'])

pd.DataFrame.from_dict(dic)  # from dictionary to pandas frame

df.info()
df.columns
df.head(10)
df.tail(10)
df.iloc[23,3]  # pick 23rd row, 3rd column
data.loc[23, 'col_name']
data.iloc[[3]]   # take 3rd recoed as dataframe because list [3] is used
df.reset_index(drop=True, inplace=True)    #reset index   
df.reset_index(drop=False)     # if index is non-integers, create new column "index"
df.fillna(0, inplace=True)   

#---------------------------------------------------------------------------

# standardize only numeric columns
for col in df.columns:
   if df[col].dtype('int64'):
      df[col] = df[col].astype('float')
binary_cols = [col for col in df.columns if df[col].dropna().isin([0, 1]).all]
float_cols = df.select_dtypes(include='float').columns.difference(binary_cols)
scaler.fit_transform(df[float_cols])

# If need to make it categorical
pd.get_dummies(input_data, columns=['xx','yyy'] )

# change column name
df.columns = ['xx','yy'].copy()    # use copy() when you see warning "value trying to be set on copy of dataframe ..."
df=df.rename(columns = {'two':'new_name'})   # change only specific column names.  

# apply moving average to particuar column using index notation
df['pandas_SMA_3'] = df.iloc[:,1].rolling(window=3).mean()

# create new column with integer 1, 2, 3 ...   
data['new_col'] = data['col'].rank(pct=True, ascending=True)  # assign integer. if pct=True, it will convert to percentage

#change data type
table[clist] = table[clist].astype(float)

# datetime parser
from dateutil import parser
dt = parser.parse("7:19 AM on March 9th")
df['col'].apply(lambda x: parser.parse(x))  # for pandas

# convert pandas column to datetime format
table['col'] = pd.to_datetime(table.datetime_col )   # include "utc=True" option to set it UTC  (add .dt.date to convert to day)
df['Time'] = df['Time'].astype('datetime64[ns]')   # if you end up with errors "Cannot compare tz-naive and tz-aware"
df['utc_datetime'].dt.date   # convert to date 
df['created_time'].dt.strftime('%Y-%m')   # convert year-month format
df['event_occurrence'].dt.to_period('M')  # aggregate dat to monthly "M".  

# Mapping using dictionary to add column.
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})

# iterate over rows
for i, row in data.iterrows():

# Get correlation between columns
combined_df['Open'].corr( combined_df['sentiment'])

# List duplicate records.
pd.concat( df for _, df in pd_ttl_data.groupby("utc_datetime") if len(df) > 1)
df[ids.isin(ids[ids.duplicated()])].sort_values("ID")  # another way
df[ df.duplicated(subset=['col']) ]  # easier way

# set column to index.
df.set_index( 'utc_datetime', inplace=True ) 
df.reset_index(drop=True, inplace=True )

# Replace "datetime" with the appropriate column from your DataFrame
df.set_index( pd.DatetimeIndex(df["datetime"]), inplace=True )

# To convert from numpy.datetime64 to datetime
pd.to_datetime(x).to_pydatetime()   
pd.to_datetime(x).to_pydatetime(tzinfo=None)  # remove timezone.

# Get column with .get()  If the value is absent, replace with 0.0  
df.get( "column_name", 0.0 )

# aggregate with Grouper.
df.groupby( pd.Grouper(key="utc_datetime", freq="1W") ).mean()

# Get stats.
df['colA'].describe()

# Chain multiple function in sequence.
df.pipe(func1).pipe(func2)

# pivot table.
df2 = df2.pivot( index='Year', columns='State', values='PacksPerCapita' )
pd.melt(df, id_vars=["student"], value_vars=["math", "science", "english"],
    var_name="subject",         # Name for the new variable column
    value_name="score"          # Name for the new value column
   )

# Create histogram dataframe.
histogram = pd.cut( df[col], bins=[0, 10, 20, float('inf')], right=False)  # bin range, [0,10) [10, 20) ...
histogram_df = pd.value_counts(histogram).reset_index()
histogram_df.columns = ['bin','freq']
 
# Append rows.
df = pd.concat( [df, row], igonore_index=True )
  
# Use groupby into lists.
df.groupby(['col']).agg( {'colX': lambda x: list(x)} )

# Use of query()
df.query('col=="A"')
df.query('col==@var') # if using variable. var='A'

# compare df.
df1 == df2
df.equals(df2)
df.compare(df2)
df1[ ~df1.apply(tuple, axis=1).isin(df2.apply(tuple, axis=1))]  # rows that exist in df1 but not in df2.
zzz.merge(xxx, how="outer", indicator=True).query('_merge == "left_only"').drop(columns="_merge")  # if zzz has extra rows, find it.
zzz[~zzz['id'].isin(xxx['id'])]  # if zzz has extra rows, find it(filter columns)

df = pd.read_csv(
    "orders.csv",
    dtype={'colA':'int16'},
    usecols=["date", "total_amount", "product"],
    parse_dates=["date"],
    infer_datetime_format=True
    )

# Use of agg()
df.groupby("region").agg(total_sales=("sales", "sum"),avg_price=("price", "mean"))

# Efficient pandas trick.
#change datatype to float32. Use category type for groupby
df.query(" colA>20 and colB<40", engine="numexpr")
# use index column when merging.
df.groupby(['col'], sort=False)  # use sort=False

# Use fugue library to run spark, ray, ... using pandas.

# multi threading pandas 
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

#-----------------------------------------------------------------------------------------------------------------

# select rows with condition
x = data['base_date'].max()  #apply aggregation to column
df[ (df['realized_pnl']>0) & (df['instrument'].isin(['USDJPY'])) | (df['utc_datetime'].isnull()) ]   # note the usage of .isin for list
df.query(f"{asset_col} == '{asset}'")  # https://note.nkmk.me/python-pandas-query/
df_group = df[ ....(condition)... ].groupby(['instrument'],as_index=False).mean()   #  mean, sum,  count, size, max, min, first(used w/ groupby), last(used w/ groupby), std, head, tail
df_group = df[ ... ].groupby(['instrument'],as_index=False).mean().loc['USDJPY','realized_pnl']   # "as_index=False" means group key will not be used as index.  
df_group = df[ ... ].groupby(['instrument'],as_index=False).size()   # .size() can be used for counting, but be warned that it will create series, not dataframe. see below. (or use 'count' instead of 'size')  
df_group = df[ ... ].groupby(['city', 'food'], as_index=False).apply(lambda x: (x.price*x.quantity).sum()  ) .to_frame('new_col').reset_index()  # Note you'll need as_index=False. 
df['new_col'] = df[ ... ].groupby(['city', 'food'], as_index=False).apply(lambda x: (x.price*x.quantity).sum()  ) .values
        # use of agg
df_group = df.groupby('city').agg({'price': myfun, 'quantity': np.sum})   # def myfun(x): ...  return x
df_group = df.groupby('A').agg(['min', 'max'])
# below will produce series (not dataframe).  need to convert from series to dataframe
df = df[ .... ].groupby(['agent_id'], as_index=False).size().to_frame('new_col_name').reset_index()   
# extract entire columns with groupby and min() condition
x = decisions.loc[ decisions.groupby(['instrument','day'])['utc_datetime'].idxmin(), : ]


# update
df['col1'] = pd.to_datetime( df.col1 )  #convert to datetime
df['new'] = df[['agent_id','instrument']].apply(lambda x: x[0]+str(x[1]), axis=1)   #add new column.  axis=1 for column.  (note: you may not need "axis=1" if there is only one column)
df['new'] = df['realized_pnl'].apply(lambda x:   100 if x[0]<0 else 0)  # add new column with if condition  (note: you may not need "axis=1" if there is only one column)
df['cumsum'] = df.groupby(['col1'])['val_col'].apply(lambda x: x.cumsum())       #apply cumsum with groupby
df['cumsum'] = df.groupby(['col1'], as_index=False)['val_col'].apply(lambda x: x.cumsum())       # set as_index=False if not working
df['cumsum'] = df.groupby(['name'])['no'].cumsum()
df['new'] = df.iloc[:,3:5].mean(axis=1)    # take average of different columns values
df.loc[ (df['utc_datetime']>time),  'colx' ] = last_order  # update with condition

# delete
df.drop(columns=['B', 'C'], axis=1)  
df.drop([3,4], axis=0)   # delete 3rd, 4th rows 
df.drop(  df[df['instrument']=='USDJPY'].index , inplace=True )  # delete with condition
df.reset_index(drop=True, inplace=True)   # usually need to fix index after removing rows 
df.drop_duplicates(subset =['col1','col2'], keep="last", ignore_index=True, inplace=True)  # drop rows from duplicate value in column 

# join
pdf_pnl5 = pd.merge(df, df, on='instrument', how='left')  #join.  left=left_join, inner=inner_join
df_new = pd.concat( [df1, df2], axis=0,  ignore_index=True )       # join at the bottom
pd.merge_asof(price_df, rma_df, left_index=True, right_index=True)   # join with nearest match (not exact match) https://qiita.com/satokiyo_DS/items/5844f697cc49258b638b
pd.merge_asof(df_left, df_right, on='a', direction='nearest')   # direction='nearest', 'backward', ... 

# convert to numpy array, or list
df.column1.tolist()
df.column1.values()
df.column1.to_numpy()








