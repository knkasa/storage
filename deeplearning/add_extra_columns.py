import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten
from tensorflow.keras.models import Model

# Setup neural network with LSTM.
input_net1 = Input(shape=(None,2,))    
hidden_net1 = LSTM(5, name="lstm_layer")(input_net1)     
output_net1 = Dense(3, use_bias=True)(hidden_net1)    
model1 = Model(inputs=input_net1, outputs=output_net1)

w1 = model1.get_layer(name="lstm_layer").get_weights()  
# There are 3 weights.  W, U, b.  See wikipedia.
# number of elements for W is (input x units x 4) (4=forget, input, output, cell weights)
# number of elements for U is (units x units x 4)

# setup 2nd network.
input_net2 = Input(shape=(None,4,))    
hidden_net2 = LSTM(7, name="lstm_layer")(input_net2)     
output_net2 = Dense(3, use_bias=True)(hidden_net2)    
model2 = Model(inputs=input_net2, outputs=output_net2)

w2 = model2.get_layer(name="lstm_layer").get_weights()

# Reshape weights. 
#w1[0].resize( np.shape(w2[0]) )
w1_size = w1[0].shape 
u1_size = w1[1].shape
b1_size = w1[2].shape
w2_size = w2[0].shape 
u2_size = w2[1].shape
b2_size = w2[2].shape

w1[0] = np.pad( w1[0], ((0, w2_size[0]-w1_size[0]), (0, w2_size[1]-w1_size[1])), constant_values=0 )
w1[1] = np.pad( w1[1], ((0, u2_size[0]-u1_size[0]), (0, u2_size[1]-u1_size[1])), constant_values=0 )
w1[2] = np.pad( w1[2], (0, b2_size[0]-b1_size[0]), constant_values=0 ) 
#import pdb; pdb.set_trace()  


# assign weights to network2.
model2.get_layer(name="lstm_layer").set_weights( weights=[ w1[0], w1[1], w1[2] ] )

print( model2.get_layer(name="lstm_layer").get_weights()[0] ) 


# below is for saving weights
temp_dir = "C:/my_working_env/ai_file_modify/weights"  
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model2)
tmp_ckpoint_tf2 = tf.train.CheckpointManager(
    checkpoint=ckpt,
    directory=temp_dir,   
    max_to_keep=1,
    keep_checkpoint_every_n_hours=None,
    checkpoint_name='model2'  ) 
#ckpt.restore( tmp_ckpoint_tf2.latest_checkpoint )   #for loading weights
#network.model = ckpt.model                   #update network just in case
tmp_ckpoint_tf2.save(checkpoint_number=100 )

model1 = model2
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model1)
tmp_ckpoint_tf2 = tf.train.CheckpointManager(
    checkpoint=ckpt,
    directory=temp_dir,   
    max_to_keep=1,
    keep_checkpoint_every_n_hours=None,
    checkpoint_name='model1'  ) 
tmp_ckpoint_tf2.save(checkpoint_number=100 )


ckpt.restore( tf.train.latest_checkpoint(temp_dir) )
model2.model = ckpt.model 
print( model2.get_layer(name="lstm_layer").get_weights()[0] )


import pdb; pdb.set_trace()  




