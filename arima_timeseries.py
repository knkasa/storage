import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import matplotlib.dates as dates
import datetime as dt
from datetime import datetime
import matplotlib.dates as mdates
import math
import itertools
import pandas.tseries.offsets as offsets
import jpholiday
import pdb

url = 'https://covid19.mhlw.go.jp/public/opendata/newly_confirmed_cases_daily.csv'

df = pd.read_csv(url, parse_dates=[0])
df = df[df["Date"] <= dt.datetime(2022, 11, 30)]
df_tokyo = df[["Date", "Tokyo"]]

df_tokyo["weekday"] = df_tokyo["Date"].dt.weekday

df_tokyo["holiday"] = False

list_holiday = []
for i in range(df_tokyo.shape[0]):
  date = df_tokyo.iloc[i,0]
  is_holiday = jpholiday.is_holiday(date)
  list_holiday.append(is_holiday)

df_tokyo["holiday"] = list_holiday

df_tokyo["holiday"].mask(df_tokyo["weekday"] == 5, True, inplace=True)
df_tokyo["holiday"].mask(df_tokyo["weekday"] == 6, True, inplace=True)

del df_tokyo["weekday"]

# 分割日 mdayの設定 (最終日から14日前)
mday = df_tokyo['Date'].iloc[-1] - offsets.Day(14)

# 訓練用indexと検証用indexを作る
train_index = df_tokyo['Date'] <= mday
test_index  = df_tokyo['Date'] > mday

# 入力データの分割
x_train = df_tokyo[train_index]
x_test  = df_tokyo[test_index]

x_train    = x_train.set_index('Date')
ts_holiday = x_train['holiday']
ts         = x_train['Tokyo']

x_test          = x_test.set_index('Date')
ts_holiday_test = x_test['holiday']

#-------- Plotting -----------------------------

# 周期性の確認。トレンドとの差分をresで取得
res = sm.tsa.seasonal_decompose(ts, period=14)
fig = res.plot()
fig.set_size_inches(16,8)
fig.tight_layout()

# 日付を絞ってプロっと
ts_check  = ts.loc['2022/10/1':'2022/11/1']
res_check = sm.tsa.seasonal_decompose(ts_check)
fig    = res_check.plot()
fig.set_size_inches(16,9)
fig.tight_layout()

# AutoCorrelation
_, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 12))

# 原系列の ACF
sm.tsa.graphics.plot_acf(ts, ax=axes[0][0])
# 原系列の PACF
sm.tsa.graphics.plot_pacf(ts, ax=axes[1][0])

# 残差の ACF
sm.tsa.graphics.plot_acf(res.resid.dropna(), ax=axes[0][1])
# 残差の PACF
sm.tsa.graphics.plot_pacf(res.resid.dropna(), ax=axes[1][1])

# 1次の階差系列の ACF
sm.tsa.graphics.plot_acf(ts.diff(1).dropna(), ax=axes[0][2])
# 1次の階差系列の PACF
sm.tsa.graphics.plot_pacf(ts.diff(1).dropna(), ax=axes[1][2])

#---------- Arima -------------------------------------

mod = sm.tsa.statespace.SARIMAX(endog = ts,
                                exog  = ts_holiday,
                                order = (4,1,2),
                                seasonal_order = (1,1,1,7),  # ７は周期性＝1週間みたいなので。
                                enforce_stationarity = False,
                                enforce_invertibility = False)
results = mod.fit()

plt.show()




pdb.set_trace()


