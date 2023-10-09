# typical example of tensorflow-keras 2.0
# https://keras.io/guides/functional_api/
# GradientTape https://qiita.com/propella/items/5b2182b3d6a13d20fefd
# batch = taking portion of input data (eg. there are N=100 sample input data.  batch means taking 20 of those data.  Durig loss calculation, only summing 20 of those.)
# loading weights-bias example.   https://www.tensorflow.org/guide/checkpoint
# A3c example.  https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
# stack multiple lstm layer https://stackoverflow.com/questions/40331510/how-to-stack-multiple-lstm-in-keras
# tf.session & tf.placeholder are gone in 2.0.   https://data-analysis-stats.jp/%E6%B7%B1%E5%B1%9E%E5%AD%A6%E7%BF%92/tensorflow-2-0-%E4%B8%BB%E3%81%AA%E5%A4%89%E6%9B%B4%E7%82%B9/
# tensorflow 1.0, session command explained.  https://qiita.com/rindai87/items/4b6f985c0583772a2e21

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

os.chdir('C:/Users/ken_nakatsukasa/Desktop/deeplearning_practice/')

# note "keras" is equivalent to tf.keras (if you didn't import keras)
# also, "layers" is equivalent to tf.keras.layers ( if you didnt import layers)
# use  var.shape,  var.numpy() , var.ndim  

#------------ defining model network ------------------------------------

inputs = tf.keras.Input(shape=(784,))   # define input layer
#img_inputs = keras.Input(shape=(32, 32, 3))   # input layer for image
print(  inputs.shape)
layer_val = tf.keras.layers.Dense(64, activation="relu")(inputs)   # 1sgt hidden layer  
tf.keras.layers.Dropout(0.2)    # drop outging connection to 2nd hidden layer by 20%
layer_val = tf.keras.layers.Dense(64, activation="relu")(layer_val)  # 2nd hidden layer
outputs = tf.keras.layers.Dense(10)(layer_val)     # output layer
# outputs = tf.keras.layers.Dense(1)(layer_val)    # used this for if output is continuous (regression)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="xxxx_model")
model.summary()  # num_param = (# unit)*(# unit/input previous layer) + (# unit)    (number of weight and bias)

#----------- getting dataset ----------------------------------------

# getting dataset.  shape is an image of 28x28 pixels.  mnist is image of hand written digits)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # input data in numpy format

x_train = x_train.reshape(60000, 784).astype("float32") / 255   # need to normalize it. 28x28=784
x_test = x_test.reshape(10000, 784).astype("float32") / 255

#------------- define loss func/optimizer ------------------------

# Loss=BinaryCrossEntropy, CategoricalCrossEntropy, MSE
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   
    optimizer=tf.keras.optimizers.RMSprop(),   # adam, SGD(stochastic grad discent), RMSProp
    metrics=["accuracy"],
)

#----------------- train the model --------------------------------

# batch_size= # of input data (batch=64) used for training.
# it uses the first 64 input data to train.  Next, it uses another 64 input data to train.  This is more memory efficient.  
res = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)    
#res = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=5)

#----------------- getting weight/bias ---------------------------

print( model.layers )  
print("Weight/bias = ", model.layers[1].get_weights()  ) #There two arrays.  1st array is weights matrix.  2nd array is bias vector.
#model.layers[1].get_weights()[1][2,5]  #e.g.  if you want particular element of weight matrix

#----------------- plotting result -------------------------------

plt.plot(res.history['loss'], label='loss' )
plt.plot(res.history['val_loss'], label='var_loss' )     # validation loss needs to be minimized
plt.legend()
plt.show()

plt.plot(res.history['accuracy'], label='accuracy' )
plt.plot(res.history['val_accuracy'], label='val_accuracy' )
plt.legend()
plt.show()

print(model.evaluate(x_test, y_test) )
print("prediction = ", model.predict(x_test)  )

#---------------- saving model ------------------------------------

model.save('./')
del model
model = tf.keras.models.load_model('./')

'''
#------------- reset weights -------------------------
# reset lstm layer weights
model_lstm = self.model.get_layer(name="lstm_layer")
w_c, w_h, bias = model_lstm.get_weights()
model_lstm.set_weights(weights=[
        model_lstm.kernel_initializer(shape=w_c.shape),
        model_lstm.kernel_initializer(shape=w_h.shape),
        model_lstm.bias_initializer(shape=len(bias)),
        ])
        
# reset policy layer weights
model_policy = self.model.get_layer(name="policy_layer")
w_policy = model_policy.get_weights()
model_policy.set_weights(weights=[model_policy.kernel_initializer(shape=w_policy[0].shape)])  # no bias term

# reset value layer weights
model_value = self.model.get_layer(name="value_layer")
w_val = model_value.get_weights()
model_value.set_weights(weights=[model_value.kernel_initializer(shape=w_val[0].shape)])  # no bias term
'''


