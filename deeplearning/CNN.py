import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

#=============================================================
# Example of convolutional neural network classification.
#=============================================================

# Load in the data.  28x28 pixels
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Plot image.
plt.imshow( x_train[0,:,:], cmap='gray' )
plt.show()

# Normalize pixel values between 0.0~1.0
x_train, x_test = x_train/255.0, x_test/255.0  

# The data is only 2D!
# CNN expects 3 dimension ( pixel x pixel x color )
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Count the number of unique target.
K = len( np.unique( y_train ) )

# Build the model.
input_net = Input( shape=(28,28,1) )
x = Conv2D( 32, (3, 3), strides=2, activation='relu')(input_net)
x = Conv2D( 64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D( 128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense( 512, activation='relu')(x)
x = Dropout(0.2)(x)
output_net = Dense(K, activation='softmax' )(x)
model = Model( inputs=input_net, outputs=output_net )

# Define loss function. 
# https://www.tensorflow.org/api_docs/python/tf/keras/losses
my_loss = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False )  # False seems better predicted.  

# Define optimizer.
#my_optimizer = tf.keras.optimizers.RMSprop( learning_rate=0.001 )
my_optimizer = tf.keras.optimizers.Adam(  )

# Define metrics.  Note that only need to define metrics for classification problem.  
# https://www.tensorflow.org/api_docs/python/tf/keras/metrics
my_metrics = tf.keras.metrics.SparseCategoricalAccuracy( )   

# Compile the loss, optimizer, metrics.
model.compile( optimizer=my_optimizer, loss=my_loss, metrics=[my_metrics] )

# Train
# If validation loss goes up, likely overfitting.  Use maxPool2D, add more data (by rotating, etc).
res = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

# Plot loss per iteration
plt.plot(res.history['loss'], label='loss')
plt.plot(res.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plot accuracy per iteration
plt.plot(res.history['sparse_categorical_accuracy'], label='acc')
plt.plot(res.history['val_sparse_categorical_accuracy'], label='val_acc')
plt.legend()
plt.show()

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))

import pdb;  pdb.set_trace()  



