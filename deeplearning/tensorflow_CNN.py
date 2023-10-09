# pooling refers to reducing dimension of image.  max-pooling means taking max value from the partition.
# stride refers to how many steps to skip when evaluating convolution
# use either pooling or stride
# Conv2d is used for image.  Conv1d is used for time series data, etc.
# batch normalization = taking portion of data for normalizing  ( it does not throw out like dropout)

import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

# getting data.  The goal is to classify 10 kinds of fashion (shoes, pants, etc)
fashion_mnist = tf.keras.datasets.fashion_mnist

# image is 28x28 size
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape:", x_train.shape)

# the data is only 2D!  convolution expects height, width, color dimension 
x_train = np.expand_dims(x_train, -1)    
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)

# number of classes to classify
K = len(set(y_train))
print("number of classes:", K)


i = Input(shape=x_train[0].shape)
# 32 is the # of filters, strides=2 means 28x28x1 becomes 14x14x32 (32 is the # of filters)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)  # (3,3) is the # of filters
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)  
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(inputs=i, outputs=x)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15, batch_size=32)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()