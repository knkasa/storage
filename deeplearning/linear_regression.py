import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler

#================================
# Example of linear regression.
#================================

N = 200  # data points
D = 2  # column

np.random.seed(seed=10)
tf.random.set_seed(10)

# create dataset X, Y
X = np.linspace( 0, 10, N*2 )
X = np.reshape( X, (N,D) )   # shape is NxD
w = np.array( [[0.25],[0.5]] )     # shape is Dx1
b = 1.
Y = np.matmul( X, w) + b + np.random.randn(N, 1) * 0.4
X[:,1] = X[:,1]*10.0  # let's make the 2nd column x*10
plt.scatter( X[:,0], Y )
plt.show()

# Split data.
x_train = X[0:int(N*0.7)]
y_train = Y[0:int(N*0.7)]
x_test = X[int(N*0.7):]
y_test = Y[int(N*0.7):]

# Normalize input/target data.
# https://stackoverflow.com/questions/38058774/scikit-learn-how-to-scale-back-the-y-predicted-result
scalerX = StandardScaler().fit(x_train)
scalerY = StandardScaler().fit(y_train)
x_train = scalerX.transform(x_train)
y_train = scalerY.transform(y_train)
x_test = scalerX.transform(x_test)
y_test = scalerY.transform(y_test)

# Note you could use scikit-learn to normalize input data.
#ct = make_column_transformer(
#    (MinMaxScaler(), ["age", "bmi", "children"]), # get all values between 0 and 1
#    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])


# Build network.
input_layer = tf.keras.layers.Input( shape=(D,) )  # If input data is NxD, let shape=(D,) where D=# of columns. If input data is NxDxM, let shape=(D,M).
output_layer = tf.keras.layers.Dense( 1, )(input_layer)  
model = tf.keras.Model( inputs=input_layer, outputs=output_layer, name="xxxx_model")
print( model.summary()  )

# You can check the output values of each network.
# Note if the input shape is NxD and the dense layer has M units, the output shape is NxM.  
# If 3D, NxDxL input shape becomes NxDxM after going through the dense layer.
tf.keras.layers.Dense( 1, activation='sigmoid')( np.random.rand(1,D) ) # Note that input shape cannot be 1D.
model.get_layer( name='dense').call( tf.constant (np.random.rand(1,D))  )

# Define loss, optimizer.  (no need to define metrics for regression problem)
my_loss = tf.keras.losses.MeanSquaredError( )  
my_optimizer = tf.keras.optimizers.RMSprop( learning_rate=0.001 )
#my_metrics = tf.keras.metrics.Accuracy()   

# Compile the loss, optimizer.
model.compile( optimizer=my_optimizer, loss=my_loss )
model.__dict__.keys()

#----------- Add callback option during training --------------------------------------------
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

output_file = "C:/my_working_env/deeplearning_practice/regression_model/best_regression_model.h5"
mc = tf.keras.callbacks.ModelCheckpoint( output_file, 
                    monitor='val_loss', mode='min',  verbose=0, save_best_only=True)

# Reduce learning rate while training.
rlp = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)

# https://keras.io/api/callbacks/
def get_lr(epoch, lr):
    if epoch>50:
        print("change learning rate = ", lr) 
        return 0.0005
    return 0.001
scheduler = tf.keras.callbacks.LearningRateScheduler(get_lr)

# Train the model.
max_epoch = 1000
res = model.fit( x_train, y_train, validation_data=(x_test, y_test), 
                epochs=max_epoch, callbacks=[mc, es, scheduler] )
res.__dict__.keys()

#------- add callback option if you want to alter learning rate --------------------------------
#-----------------------------------------------------------------------------------------------

# Plot loss.
plt.plot(res.history['loss'], label='loss')
plt.plot(res.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Need to denormalize back.
y_pred0 = model.predict(x_test)    # model( x_test ) will output in tensor format.  
y_pred = scalerY.inverse_transform(y_pred0)
y_target = scalerY.inverse_transform(y_test)
x_test = scalerX.inverse_transform(x_test)

# Plot result
plt.scatter( x_test[:,0], y_target, label='target' )
plt.plot( x_test[:,0], y_pred, label='prediction' )
plt.show()

# save/load model.
#model.save('logistic_model.tf')
#model = tf.keras.models.load_model('logistic_model.tf')
best_model = tf.keras.models.load_model( output_file  )
model = best_model

#import pdb; pdb.set_trace() 



