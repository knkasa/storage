import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#=======================================
# Example of logistic regression.
#=======================================

os.chdir('C:/Users/ken_nakatsukasa/Desktop/deeplearning_practice/')

# Load the data.  Data is like dictionary.
data = load_breast_cancer()
print( data.keys() )

data.data  # input data 
data.target  # prediction data 
data.target_names  # prediction names
data.feature_names  # input column names

# Split data
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
N, D = np.shape( X_train )

# Normalize input data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Note you could use scikit-learn to normalize input data.
#ct = make_column_transformer(
#    (MinMaxScaler(), ["age", "bmi", "children"]), # get all values between 0 and 1
#    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])

# Build network. Use Activation=Softmax
# Note if the input shape is NxD and the dense layer has M units, the output shape is NxM.  
# If 3D, NxDxL input shape becomes NxDxM after going through the dense layer.
input_layer = tf.keras.layers.Input( shape=(D,) )  # If input data is NxD, let shape=(D,) where D=# of columns. If input data is NxDxM, let shape=(D,M).
output_layer = tf.keras.layers.Dense( 1, activation='sigmoid', )(input_layer)  
model = tf.keras.Model( inputs=input_layer, outputs=output_layer, name="xxxx_model")
print( model.summary()  )

# You can check the output values of each network.
tf.keras.layers.Dense( 1, activation='sigmoid')( np.random.rand(1,D) )
model.get_layer( name='dense').call( tf.constant (np.random.rand(1,D))  )

# Define loss function. 
# https://www.tensorflow.org/api_docs/python/tf/keras/losses
# For multiclass classification use CategoricalCrossEntropy()
#my_loss = tf.keras.losses.BinaryCrossentropy( from_logits=True )  # set logits=True if output_net is sigmoid or softmax. 
my_loss = tf.keras.losses.BinaryCrossentropy( from_logits=False )  # False seems better predicted.  

# Define optimizer.
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
my_optimizer = tf.keras.optimizers.RMSprop( learning_rate=0.001 )
#my_optimizer = tf.keras.optimizers.Adam(  )

# Define metrics.  Note that only need to define metrics for classification problem.  
# https://www.tensorflow.org/api_docs/python/tf/keras/metrics
my_metrics = tf.keras.metrics.BinaryAccuracy( threshold=0.5 )   

# Compile the loss, optimizer, metrics.
model.compile( optimizer=my_optimizer, loss=my_loss, metrics=[my_metrics] )
model.__dict__.keys()

# Note: use tf.kerass.callbacks.LearningRateScheduler() to vary learn rate to check to what is the best LR.  

# Train the model.
#res = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)
#res.__dict__.keys()

#----------- Add callback option during training --------------------------------------------
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

output_file = "C:/my_working_env/deeplearning_practice/regression_model/best_regression_model.h5"
mc = tf.keras.callbacks.ModelCheckpoint( output_file, 
                    monitor='val_loss', mode='min',  verbose=0, save_best_only=True)
                    
# Reduce learning rate while training.
rlp = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)

#------- add callback option if you want to alter learning rate --------------------------------
# https://keras.io/api/callbacks/
def get_lr(epoch, lr):
    if epoch>50:
        print("change learning rate = ", lr)
        return 0.0001
    return 0.001
scheduler = tf.keras.callbacks.LearningRateScheduler(get_lr)
res = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[es, mc, scheduler])
res.__dict__.keys()
#-----------------------------------------------------------------------------------------------

# Evaluate the model - evaluate() returns loss and accuracy
print("Test score = ", model.evaluate(X_test, y_test))

# Plot loss.
plt.plot(res.history['loss'], label='loss')
plt.plot(res.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plot the accuracy .
plt.plot(res.history['binary_accuracy'], label='test_accuracy')
plt.plot(res.history['val_binary_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# Test result.  
pred_res10 = model.predict( X_test[:10] )   # model( X_test[:10] ) will output in tensor format.  
print("Prediction = ", np.round(pred_res10).flatten() )  # Need to round numbers.  
print("Target = ", y_test[:10] )

# save/load model.
#model.save('logistic_model.tf')
#model = tf.keras.models.load_model('logistic_model.tf')
best_model = tf.keras.models.load_model( output_file  )
model = best_model

#import pdb; pdb.set_trace()  


