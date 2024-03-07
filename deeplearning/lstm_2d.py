import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

#=============================================================
# Example of LSTM 2d time-series predictions.  
#=============================================================

# Predicting a simple sine function.  Try to vary T.
# T = the number of input time step for predicting the next value.

T = 5    # send T past values, and predict the next value
D = 2   # number of input column

#------------- Preparing data ---------------------------------------------

time = np.linspace( 0, 5*np.pi, 201 ) 
out = np.sin( time )
out2 = np.cos( time )

plt.plot( time, out, 'o-', time, out2, 'o-' )
plt.show() 

#--------- Prepare sets of data (NxTxD) -------------------------------------

out = np.reshape( out, [-1,1] )   # column 1
out2 = np.reshape( out2, [-1,1] )   # column 2
out3 = np.concatenate( (out, out2), axis=1 )   # combine two columns

X=[]
Y=[]
for t in range(len(out) - T):
    x = out3[t:t+T]
    X.append(x)
    y = out3[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T, D)   # make it NxTxD
Y = np.array(Y)

N = len(X)

# Note X[0] x[1] ... are older input.  x[-1] is the newest input.
print(" Send X[0,:,:] = " )
print( X[0,:,:] )
print(" and predict the next value Y[0] = ")
print( Y[0] )  
print(" Expected result should be ")
print( out3[0:T+1,:]   );  print() 

#-------------------- Now try LSTM model -----------------------------------------

# Setup neural network with LSTM.
input_net = Input(shape=(None,D,))  #shape=(T, D,) works.  None means it accepts any dimension
hidden_net = LSTM(20)(input_net)     
output_net = Dense(D)(hidden_net)    
model = Model(inputs=input_net, outputs=output_net)

# Define loss.
# https://www.tensorflow.org/api_docs/python/tf/keras/losses
my_loss = tf.keras.losses.MeanSquaredError()

# Define optimizer.
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
my_optimizer = tf.keras.optimizers.Adam( clipvalue=0.5 )


# Compile the model.
model.compile( loss=my_loss, optimizer=my_optimizer )
model.__dict__.keys()

# Train the RNN.  batch_size=# of samples to calculate loss function.  batch_size<N(train)
res = model.fit( X[:-N//2], Y[:-N//2], batch_size=32, epochs=100, validation_data=( X[-N//2:], Y[-N//2:])  )

# Plot some data
plt.plot(res.history['loss'], label='loss')
plt.plot(res.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

#------------- Result -------------------------------------------------------------

forecast = []
input_test = X[-N//2]  # use the validation data to test

while len(forecast) < len(Y[-N//2:]):

	# Reshape the input_ to N x T x D
	x = model.predict( input_test.reshape(1, T, D) )  # remember to reshape 1xTxD
	forecast.append(x)
	input_test = np.concatenate( (input_test, x), axis=0 )  # append the forcast to input data.
	input_test = np.delete( input_test , 0 , 0 )  # delete the first row of input.

predicted =  np.squeeze( np.array(forecast) )  # reduce it to 1 dimension.
	
#import pdb;  pdb.set_trace()  
plt.plot( Y[-N//2:,0], 'o-', label='targets' )
plt.plot( predicted[:,0], 'o-', label='forecast' )
plt.legend()
plt.show()
