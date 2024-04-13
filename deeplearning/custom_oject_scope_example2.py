import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import custom_object_scope
import os
import pdb

# When saving model file with .h5, and you define your own ustom network (my_model), 
# you need to call custom_object_scope to load the model.  
# Or, you can simply call restore_best_weights=True in earlyStopping.  

os.chdir("C:/Users/knkas/Desktop/NLP_example")

N = 100
D = 2
unit = 10

X = np.linspace( 0, 10, N*2 )
X = np.reshape( X, (N,D) )   # shape is NxD
w = np.array( [[0.25],[0.5]] )     # shape is Dx1
b = 1.0
Y = np.matmul(X, w) + b + np.random.randn(N, 1)*0.4

class my_model(tf.keras.layers.Layer):
    def __init__(self,unit,**kwargs):
        super(my_model, self).__init__(**kwargs)

        self.unit = unit

        #self.input = tf.keras.layers.Input(shape=(D,))
        self.dense = tf.keras.layers.Dense(unit)
        self.output_net = tf.keras.layers.Dense(1)

    def call(self, x):

        x = self.dense(x)
        x = self.output_net(x)

        return x
    
    def get_config(self):
        config = super(my_model, self).get_config()
        config.update({'unit':self.unit})
        return config

X = tf.constant(X)
Y = tf.constant(Y)

model_net = my_model(10,)

es = tf.keras.callbacks.EarlyStopping( patience=20, )  # restore_best_weights=True
mc = tf.keras.callbacks.ModelCheckpoint('/tf_model_temp/tf_model_temp.h5', monitor='val_loss', mode='min', save_best_only=True)

input_net = tf.keras.layers.Input(shape=(D))
output_net = model_net(input_net)
model = tf.keras.Model( inputs=input_net, outputs=output_net, name='xxxx')

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005)
    )

res = model.fit( X, Y, validation_split=0.3, epochs=1000, batch_size=32, callbacks=[es, mc] )

with custom_object_scope({'my_model':my_model}):
    model = tf.keras.models.load_model('/tf_model_temp/tf_model_temp.h5')



