from random import seed
from random import randrange
from matplotlib import pyplot
from random import random
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Randomnesss in time series.
# https://machinelearningmastery.com/gentle-introduction-random-walk-times-series-forecasting-python/

data = pd.read_csv("C:/my_working_env/deeplearning_practice/processed_data.csv")
price = data.PRICE_RSI1440_mean.values

# random numbers 
seed(1)
series = [ randrange(10) for i in range(100) ]
pyplot.plot(series)
pyplot.title("random numbers")
pyplot.show()

# random walk.
random_walk = list()
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1, 800):
	movement = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + movement
	random_walk.append(value)
pyplot.plot(random_walk)
pyplot.title("random walk")
pyplot.show()

# price.
pyplot.plot(price[:100])
pyplot.title("price")
pyplot.show()


#We can calculate the correlation between each observation and the observations 
#  at previous time steps. A plot of these correlations is called an autocorrelation 
#  plot or a correlogram.

autocorrelation_plot(random_walk)
pyplot.title("random walk")
pyplot.show()

autocorrelation_plot(series)
pyplot.title("random numbers") 
pyplot.show()

autocorrelation_plot(price[:500])
pyplot.title("price") 
pyplot.show()

# Running the example, we can see that the test statistic value was 0.341605.
# This is larger than all of the critical values at the 1%, 5%, and 10% confidence levels.
# Therefore, we can say that the time series does appear to be non-stationary
# with a low likelihood of the result being a statistical fluke().

# statistical test (random walk)
result = adfuller(random_walk)
print("random walk")
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
print() 

# statistical test (random numbers) 
result = adfuller(series)
print("random numbers")
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
print()

# statistical test (price) 
result = adfuller(price[:500])
print("price")
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
print()


# Determine whether the time series is random walk or not.
# take difference, then get statistics.  
diff = list()
for i in range(1, len(random_walk)):
	value = random_walk[i] - random_walk[i - 1]
	diff.append(value)

# statistical test (diff random walk each step) 
result = adfuller(diff)
print("diff random walk")
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

autocorrelation_plot(diff)
pyplot.title("diff random walk") 
pyplot.show()


# statistical test (diff price)
diff = list()
for i in range(1, len(price)):
	value = price[i] - price[i - 1]
	diff.append(value)

# statistical test (diff price) 
result = adfuller(diff)
print("diff price")
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

autocorrelation_plot(diff[:400])
pyplot.title("price diff") 
pyplot.show()




