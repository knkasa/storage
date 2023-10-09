from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example of LSTM time-series predictions.  
# Predicting a simple sine function.  Try to vary T 
# T = the number of input time step for predicting the next value)

T = 1    # send T past values, and predict the next value
D = 1   # number of input column

#------------- preparing t & ouput values ---------------------------------------------

time = np.linspace( 0, 15*np.pi, 101 ) 
out = np.sin( time )

#plt.figure(1)
#plt.plot( time, out )
#plt.show() 

#--------- build the dataset ----------------------------------------------------------

# let's see if we can use T past values to predict the next value


X = []
Y = []
for t in range(len(out) - T):
    x = out[t:t+T]
    X.append(x)
    y = out[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T)    # make it N x T.  N=# of rows (samples) to train
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)
print(" send X[0,:] = ", X[0,:], " and predict the next value Y[0] = ", Y[0] )  
print( out[0:T+1 ]   );  print() 

#-------------------- Now try LSTM model -----------------------------------------
X = X.reshape(-1, T, D) # make it N x T x D

# setup neural network with LSTM
input = Input(shape=(None, D))  #shape=(T, D) works.  None means it accepts any dimension
hidden = LSTM(10)(input)   # number of units is 10
output = Dense(1)(hidden)   
model = Model(input, output)
model.compile( loss='mse', optimizer=Adam(lr=0.02), )

# train the RNN.  batch_size=# of samples to calculate loss function.  batch_size<N(train)
res = model.fit( X[:-N//2], Y[:-N//2], batch_size=32, epochs=100, validation_data=( X[-N//2:], Y[-N//2:])  )

# plot some data
#plt.figure(2)
#plt.plot(res.history['loss'], label='loss')
#plt.plot(res.history['val_loss'], label='val_loss')
#plt.legend()
#plt.show()


#------------- result -------------------------------------------------------------

forecast = []
input_ = X[-N//2]  # use the validation data to test

while len(forecast) < len(Y[-N//2:]):
    # Reshape the input_ to N x T x D
    f = model.predict(input_.reshape(1, T, D))[0,0]  # dimension=NxTxD
    forecast.append(f)

    # make a new input with the latest forecast
    input_ = np.roll(input_, -1)
    input_[-1] = f

plt.figure(3)
plt.plot(Y[-N//2:], 'o-', label='targets')
plt.plot(forecast, 'o-', label='forecast')
plt.title("LSTM Forecast")
plt.legend()
plt.show()


