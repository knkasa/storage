# example of multiple inputs/outputs

#if you're building a system for ranking customer issue tickets by priority and routing them to the correct department, then the model will have three inputs:
#  1. the title of the ticket (text input),
#  2. the text body of the ticket (text input), and
#  3. any tags added by the user (categorical input)
#This model will have two outputs:
#  1. the priority score between 0 and 1 (scalar sigmoid output), and
#  2. the department that should handle the ticket (softmax output over the set of departments).

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_tags = 12  # Number of unique issue tags
num_words = 10000  # Size of vocabulary obtained when preprocessing text data
num_departments = 4  # Number of departments for predictions

# define 3 inputs 
title_input = keras.Input(shape=(None,), name="title")  # Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
tags_input = keras.Input(shape=(num_tags,), name="tags")  # Binary vectors of size `num_tags`

# Embed each word in the title into a 64-dimensional vector.  (remember that these inputs are texts)
title_features = layers.Embedding(num_words, 64)(title_input)
# Embed each word in the text into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)

# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128)(title_features)
# Reduce sequence of embedded words in the body into a single 32-dimensional vector
body_features = layers.LSTM(32)(body_features)

# Merge all available features into a single vector "x" via concatenation
x = layers.concatenate([title_features, body_features, tags_input])

# define two output layers
priority_pred = layers.Dense(1, name="priority")(x)
department_pred = layers.Dense(num_departments, name="department")(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(  inputs=[title_input, body_input, tags_input],  outputs=[priority_pred, department_pred],  )

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.BinaryCrossentropy(from_logits=True),
          keras.losses.CategoricalCrossentropy(from_logits=True), ],
    loss_weights=[1.0, 0.2],  )