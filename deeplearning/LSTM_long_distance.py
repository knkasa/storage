import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

#=============================================================
# Example of LSTM long distance classification problem.
#=============================================================

### build the dataset
# This is a nonlinear AND long-distance dataset
# (Actually, we will test long-distance vs. short-distance patterns)

# Label either 1 or 0.
def get_label(x, i1, i2, i3):
	# x = sequence
	if x[i1] < 0 and x[i2] < 0 and x[i3] < 0:
		return 1
	if x[i1] < 0 and x[i2] > 0 and x[i3] > 0:
		return 1
	if x[i1] > 0 and x[i2] < 0 and x[i3] > 0:
		return 1
	if x[i1] > 0 and x[i2] > 0 and x[i3] < 0:
		return 1
	return 0

# **** Start with a small T. Making T larger will be harder to predict. *****
T = 50
D = 1    # number of columns
X = []
Y = []

# Create list of numbers (each list has T values) and classify 1 or 0.
for t in range(5000):
	x = np.random.randn(T)  
	X.append(x)
	#y = get_label(x, -1, -2, -3)   # x[-1] x[-2] ... will be short distance problem.
	y = get_label(x, 0, 1, 2)     # x[0] x[1] ... will be long distance problem.  
	Y.append(y)

X = np.array(X)  # Input
Y = np.array(Y)  # Target
N = len(X)

#------------ Set up neural network -----------------------------

# Setup neural network with LSTM.
input_net = Input(shape=(None,D,))  #shape=(T, D,) works.  None means it accepts any dimension
hidden_net = LSTM(20)(input_net)     
output_net = Dense(D, activation='sigmoid')(hidden_net)    
model = Model(inputs=input_net, outputs=output_net)

my_loss = tf.keras.losses.BinaryCrossentropy( from_logits=False )
my_optimizer = tf.keras.optimizers.Adam(  )
my_metrics = tf.keras.metrics.BinaryAccuracy( threshold=0.5 )   

# Compile the model.
model.compile( loss=my_loss, optimizer=my_optimizer, metrics=[my_metrics])

# Before training, need to change dimension to NxTxD for LSTM.
inputs = np.expand_dims(X, -1)

# Train
res = model.fit( inputs, Y, epochs=50, validation_split=0.5 )

plt.plot(res.history['loss'], label='loss')
plt.plot(res.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plot the accuracy too
plt.plot(res.history['binary_accuracy'], label='acc')
plt.plot(res.history['val_binary_accuracy'], label='val_acc')
plt.legend()
plt.show()

import pdb;  pdb.set_trace()  




