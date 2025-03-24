import pandas as pd
import numpy as np
import pdb
from pandasql import sqldf
from faker import Faker

np.random.seed(0)

# Create the first dataframe
fake = Faker()
date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
df1 = pd.DataFrame({
    'date': pd.to_datetime(np.random.choice(date_range, 10000)),
    'price': np.random.uniform(10, 100, 10000),
    'ID': np.random.choice(['A', 'B', 'C'], 10000)
    })

# Create the second dataframe
df2 = pd.DataFrame({
    'ID': np.random.choice(['A', 'B', 'C'], 1000),
    'state': [fake.state() for _ in range(1000)],
    'age': np.random.randint(18, 80, 1000)
    })

'''
SELECT
    date_trunc('day', original_time) as base_date,
    sum(realized_pnl) OVER (PARTITION BY date_trunc('day', original_time), symbol) as sum_realized
FROM table;
'''

qry = " select * from df1 where date > '2022-01-01' "

qry = " with temp as ( select * from df1 where ID='A' union select * from df1 where ID='B' ) select * from temp "
qry = " select date, iif( price>90, 'high', 'low' ) as price_rank from df1 "

# Cumulative sum.
qry = " select date, ID, price, sum(price) over( partition by ID order by date asc) as cum_sum from df1 "
#qry = " select date, ID, price, rank() over( partition by ID order by date asc) as ranking from df1 "
qry = " select date, ID, price, row_number() over( partition by ID order by date asc) as trade_id from df1 "

print( sqldf( qry, globals() ) ) 

