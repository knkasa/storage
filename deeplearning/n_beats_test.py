# https://arxiv.org/pdf/1905.10437.pdf 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

num_data_points = 1001
WINDOW_SIZE = 5  # number of lookup rows for predictions. (note it needs to be columns in input data.)
HORIZON = 1   # number of rows for predictions. 

# Values from N-BEATS paper Figure 1 and Table 18/Appendix D
N_EPOCHS = 2000    # called "Iterations" in Table 18.  Default=5000
N_NEURONS = 512    # called "Width" in Table 18.
N_LAYERS = 4
N_STACKS = 30
learn_rate = 0.0001

#------------- Preparing data ---------------------------------------------

tf.random.set_seed(42)

INPUT_SIZE = WINDOW_SIZE*HORIZON     # called "Lookback" in Table 18. Note this is just number of columns in input data.
THETA_SIZE = INPUT_SIZE+HORIZON

time = np.linspace( 0, 61*np.pi, num_data_points ) 
out = np.sin( time )
plt.plot( time, out, 'o-' )
#plt.show() 
plt.close()
#exit() 

# prepare input data.  Add lookup values for each columns.
dic = { 'input':out, }
df = pd.DataFrame.from_dict(dic)
for i in range(WINDOW_SIZE):
    df[f"input{i+1}"] = df["input"].shift(periods=i+1)
X = df.dropna().drop("input", axis=1).astype(np.float32)
target = df.dropna()["input"].astype(np.float32)

split_size = int(len(X)*0.8)
X_train, y_train = X[:split_size], target[:split_size]
X_test, y_test = X[split_size:], target[split_size:]

#----------- Create NBeatsBlock layer -------------------------------- 
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self,
                    input_size: int,
                    theta_size: int,
                    horizon: int,
                    n_neurons: int,
                    n_layers: int,
                    **kwargs): 
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        # Block contains stack of 4 fully connected layers each has ReLU activation
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
        # Output of block is a theta layer with linear activation
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs): # Note "call" function is automatically called.
        
        x = inputs
        for layer in self.hidden: 
          x = layer(x)
        theta = self.theta_layer(x) 
        
        # Output the backcast and forecast from theta
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast


#----------- check dummy output ---------------------------------

# Set up NBeatsBlock layer to represent inputs and outputs.
test_block_layer = NBeatsBlock(input_size=INPUT_SIZE, 
                                theta_size=THETA_SIZE,  
                                horizon=HORIZON,
                                n_neurons=N_NEURONS,
                                n_layers=N_LAYERS,
                                name='TestBlock', 
                                )

dummy_input = X_train.head(1).values
import pdb; pdb.set_trace()

# check dummy outputs.  Note the size of backcast and forcast.  
backcast, forecast = test_block_layer(dummy_input)
print(f"Dummy backcast: {tf.squeeze(backcast.numpy())}")
print(f"Dummy forecast: {tf.squeeze(forecast.numpy())}")

#------ adding and subtracting layers (example) -----------------

# Make tensors
tensor_1 = tf.range(10) + 10
tensor_2 = tf.range(10)

# Subtract
subtracted = tf.keras.layers.subtract([tensor_1, tensor_2])

# Add
added = tf.keras.layers.add([tensor_1, tensor_2])


#------------- Make train and test sets --------------------------

'''
# Use of tf.data for input data. 
# 1. Turn train and test arrays into tensor Datasets
train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

# 2. Combine features & labels
train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

# 3. Batch and prefetch for optimal performance
BATCH_SIZE = 1024 # taken from Appendix D in N-BEATS paper
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
'''


#------------ Set up entire network -------------------------------------

# 1. Setup input layers.  
stack_input = tf.keras.layers.Input( shape=(INPUT_SIZE), name="stack_input")

# 2. Get initial backcast, forecast layers. Block1
initial_block_layer = NBeatsBlock(input_size=INPUT_SIZE, 
                                theta_size=THETA_SIZE,  
                                horizon=HORIZON,
                                n_neurons=N_NEURONS,
                                n_layers=N_LAYERS,
                                name='InitialBlock', # note this is just initial block.
                                )

backcast, forecast = initial_block_layer(stack_input)

# 3. Get the first residual layers. input minus backcast.
residual = tf.keras.layers.subtract( [stack_input, backcast], name=f"subtract_00") 

# 4. Create stacks of remaining blocks. Block2, Block3, ...
for i, _ in enumerate(range(N_STACKS-1)): 

    # 5. Use the NBeatsBlock to calculate the backcast as well as forecast.
    block_layer = NBeatsBlock(input_size=INPUT_SIZE,
                                theta_size=THETA_SIZE,
                                horizon=HORIZON,
                                n_neurons=N_NEURONS,
                                n_layers=N_LAYERS,
                                name=f"NBeatsBlock_{i}" )
                                
    backcast, block_forecast = block_layer(residual)

    # 6. Create the double residual stacking
    residual = tf.keras.layers.subtract( [residual, backcast], name=f"subtract_{i}") 
    forecast = tf.keras.layers.add( [forecast, block_forecast], name=f"add_{i}")

#-------- 7. Put the model together and compile --------------------------------------

model = tf.keras.Model( inputs=stack_input, outputs=forecast, name="model_7_N-BEATS")

model.compile(loss="mae",
                  optimizer=tf.keras.optimizers.Adam(learn_rate),
                  metrics=["mae", "mse"])

# add callback options.
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
rlp = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=15, verbose=1)

# 9. Fit the model with EarlyStopping and ReduceLROnPlateau callbacks
N = len(X_train)
res = model.fit(X_train[:int(N*0.7)],
            y_train[:int(N*0.7)],
            epochs=N_EPOCHS,
            #batch_size=64, 
            validation_data=( X_train[-int(N*0.3):], y_train[-int(N*0.3):] ) ,
            verbose=1, 
            callbacks=[es, rlp])

#--------- plot result ----------------------------

forecast = []
input_val = X_test.head(1).values
for n in range(len(X_test)):

    x_pred = model.predict( input_val, verbose=0 )  # remember to reshape 1xTxD
    forecast.append(x_pred)
    input_val = np.roll( input_val, 1 )
    input_val[0,0] = x_pred
    
  
predicted =  np.squeeze( np.array(forecast) )	 # reduce it to 1 dimension.  

plt.plot( predicted, 'o-', label='predicted' )
plt.plot( y_test.values, 'o-', label='target' )
plt.legend(loc='upper right') 
plt.show()


import pdb; pdb.set_trace()  








