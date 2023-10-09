import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

# subclass method
# https://www.tensorflow.org/guide/keras/custom_layers_and_models
# https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e

os.chdir('C:/Users/ken_nakatsukasa/Desktop/deeplearning_practice/')

# note "keras" is equivalent to tf.keras (if you didn't import keras)
# also, "layers" is equivalent to tf.keras.layers ( if you didnt import layers)
# use  var.shape,  var.numpy()   

# use either tf.keras.Model, or tf.keras.layers.Layer.
# If tf.keras.layers.Layer is used, you must close the network with tf.keras.Model( inputs=your_input, outputs=your_output, name="your_model")
# For the example of tf.keras.layers.Layer(), see n_beats.py.  
# Note if you use tf.keras.Model approach instead of tf.keras.layers.Layer, you don't define input shape here.  You define that later (see below).
class MyModel(tf.keras.Model):   
    def __init__(self, mode):
        super(MyModel, self).__init__()   # super().__init__()  is also fine too.  
		
        # Note there is no input layer
        self.dense = tf.keras.layers.Dense(8, activation="sigmoid")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(1)  # note you cannnot use self-defined name "self.output" "self.outputs"
        if mode=="mode1":
            self.dense3 = tf.keras.layers.Dense(8)
        else:
            self.dense3 = tf.keras.layers.Dense(8)

    def call(self, input, training):  # training option can be set for dropout layer below
        x = self.dense(input)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x 
	
X = np.arange(0, 100, 0.1)
y = 0.5*X + np.random.rand(len(X))
X = np.reshape(X, (len(X),1) )
y = np.reshape(y, (len(y),1) )

X = tf.convert_to_tensor( X, dtype=tf.float32 )
y = tf.convert_to_tensor( y, dtype=tf.float32 )


model = MyModel(mode="mode1")

#model.build( input_shape=(len(X),1) )  # Define input shape here.
model.build(input_shape=(None, 1))   # Define input shape here NxD D=# of columns.  For LSTM, NxTxD (None, T, D).  
print(model.summary())

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
loss = tf.keras.losses.MeanSquaredError()
#metrics = tf.keras.metrics.Accuracy()

#print( model( tf.ones(shape=(len(X),1))) )

#model.compile( optimizer=optimizer, loss=loss, metrics=metrics )
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=10, batch_size=64 )


