import pandas as pd
import numpy as np
import sqlalchemy as sa
import pandas_ta as ta
import decimal
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import talib 


# C:\Users\ken_nakatsukasa\Anaconda3\envs\myWorkspace2\lib\site-packages\talib
# https://en.wikipedia.org/wiki/Relative_strength_index
# https://github.com/bukosabino/ta/issues/38
# https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/rsi.py
# https://github.com/bukosabino/ta

mysql_engine = sa.create_engine("postgresql+psycopg2://xxx:yyy@zzz.com/price_db")   
str_query = " select  instrument, utc_datetime, bid_close, bid_high, bid_low from price_history_candlestick where instrument='USDJPY' and utc_datetime>='2022-11-01 00:00:00' order by instrument  "

df = pd.read_sql_query(str_query, mysql_engine)
df.reset_index(drop=True, inplace=True)

#help(df.ta)
#df.ta.indicators()

help(ta.macd)
#help(ta.sma)
#help(ta.rsi)
#help(ta.bbands)
#help(ta.rsi)
#help(ta.stoch)
#help(ta.ichimoku)
exit()

#df[['sar1'],['sar2'],['sar3'],['sar4']] = ta.psar(df["bid_high"], df["bid_low"], )

df['pandas_sma'] = ta.sma(df['bid_close'], length=1440)/df['bid_close'].values
df['pandas_rsi'] = ta.rsi(df['bid_close'], length=1440, scalar=100.0, drift=1, offset=0 ) 
df[['pandas_band3','pandas_band2','pandas_band1','pandas_band4','pandas_band5']] = ta.bbands(df['bid_close'], length=1440, std=2, mamode='sma').div( df['bid_close'], axis=0 )
df.drop(columns=['pandas_band4', 'pandas_band5'], axis=1, inplace=True) 

df[['pandas_stoch1','pandas_stoch2']] = ta.stoch(df['bid_high'], df['bid_low'], df['bid_close'], length=1440, k=1440, d=1440/5*3, smooth_k=1440/5*3)
df[['pandas_macdl','pandas_macdh','pandas_macds']] = ta.macd(df['bid_close'], fast=1440, slow=1440/12*26, signal=1440/12*9, offset=100)
df['pandas_env'] = df['bid_close'].values - ( ta.sma(df['bid_close'], length=1440).values/df['bid_close'].values )

print( df ) 

ichi1, ichi2 = ta.ichimoku( df['bid_high'], df['bid_low'], df['bid_close'], tenkan=1440*5, kijun=1440/9*26*5, senkou=1440/9*52*5 )
df[['ichi1','ichi2','ichi3','ichi4', 'ichi5']] = ichi1

df.to_csv('C:/my_working_env/talib_test/pandas.csv')
#ichi1.to_csv('C:/my_working_env/talib_test/pandas2.csv')


'''
#import pdb; pdb.set_trace() 
for n in range(10):
    df[['sar1','sar2','sar3','sar4']] = ta.psar( df.head((n+1)*10)["bid_high"], df.head((n+1)*10)["bid_low"], )
    #sar = pd.merge(df[['utc_datetime','bid_high','bid_close','bid_low']], sar, left_index=True, right_index=True)
    figure(n+1)
    ax = plt.gca()
    df.head((n+1)*10).plot( x='utc_datetime', y='bid_close', ax=ax ) 
    df.head((n+1)*10).plot( x='utc_datetime', y='sar1', ax=ax ) 
    df.head((n+1)*10).plot( x='utc_datetime', y='sar2', ax=ax ) 
    df.to_csv("C:/my_working_env/talib_test/check" + str((n+1)*10) + "xx.csv")

df[['sar1','sar2','sar3','sar4']] = ta.psar( df.head(1000)["bid_high"], df.head(1000)["bid_low"], )
#sar = pd.merge(df[['utc_datetime','bid_high','bid_close','bid_low']], sar, left_index=True, right_index=True)
df.to_csv("C:/my_working_env/talib_test/check" + str(1000) + ".csv")
figure(11)
ax = plt.gca()
df.head(100).plot( x='utc_datetime', y='bid_close', ax=ax ) 
df.head(100).plot( x='utc_datetime', y='sar1', ax=ax ) 
df.head(100).plot( x='utc_datetime', y='sar2', ax=ax ) 
plt.show()
#plt.close()
'''



import pdb; pdb.set_trace()  


#SAR explained.
# https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar#:~:text=Current%20SAR%20%3D%20Prior%20SAR%20%2B%20Prior,to%20the%20prior%20period's%20SAR.
# https://www.investopedia.com/terms/p/parabolicindicator.asp

