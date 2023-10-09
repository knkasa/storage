

# This code demonstrates how to update weight/bias manually  

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

os.chdir('C:/Users/ken_nakatsukasa/Desktop/deeplearning_practice/')

inputs = tf.keras.Input(shape=(784,), name='input_layer' )   # define input layer
layer_val = tf.keras.layers.Dense(10, activation="relu", name='hidden_layer' )(inputs)   # 1sgt hidden layer  
outputs = tf.keras.layers.Dense(1, name='output_layer')(layer_val)     # output layer

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="xxxx_model")
model.summary()

optimizer=tf.keras.optimizers.RMSprop()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # input data in numpy format
x_train = x_train.reshape(60000, 784).astype("float32") / 255   # need to normalize it
x_test = x_test.reshape(10000, 784).astype("float32") / 255

tf_test = tf.convert_to_tensor( y_test,  dtype=tf.float32)   # data might need to be in tensorflow format

# all the calculation needs to be inside GradienTape
with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:

    # choose tape.watch to define which variable to take derivative
    tape.watch(model.get_layer("hidden_layer").trainable_variables)
    tape.watch(model.get_layer("output_layer").trainable_variables)

    # Here, you could also use "output_value = self.model(x_test)"
    input_value = model.get_layer("input_layer", index=None).call(x_test)
    hidden_value = model.get_layer("hidden_layer", index=None).call(input_value)
    output_value = model.get_layer("output_layer", index=None).call(hidden_value)
    
    loss = tf.reduce_mean(tf.abs(tf_test - tf.squeeze(output_value)  )  )


xgrad = tape.gradient(
                    target=loss,
                    sources=model.trainable_variables,
                    output_gradients=None,
                    unconnected_gradients=tf.UnconnectedGradients.NONE
                )
                
optimizer.apply_gradients(zip(xgrad, model.trainable_variables))
