import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.utils import shuffle
import os

#=============================================================
# Example of Recommender system.  (Movie rating)
# You want to predict similarities in your data.  
# Need separate embedding for each variable.
#=============================================================

#-----------------------------------------------------------------
# Get input data

os.chdir("C:/Users/ken_nakatsukasa/Desktop/deeplearning_practice")

df = pd.read_csv('movie_rating.csv')

# Assign User ID using Categorical (assign ID from 0 to last id) (no missing number in between)
df.userId = pd.Categorical(df.userId)
df['new_user_id'] = df.userId.cat.codes

# Assign Movie ID
df.movieId = pd.Categorical(df.movieId)
df['new_movie_id'] = df.movieId.cat.codes

# Get user IDs, movie IDs, and ratings as separate arrays.
user_ids = df['new_user_id'].values   # input data 
movie_ids = df['new_movie_id'].values  # another input data
ratings = df['rating'].values

# Get number of unique users and number of unique movies.
N = len(set(user_ids))
M = len(set(movie_ids))

#import pdb;  pdb.set_trace()  

#---------------------------------------------------------------------
# Create model

# Set embedding dimension in neural network.
K = 10

# Input network
user_net = Input( shape=(1,) )  # Only one column for both input data
movie_net = Input( shape=(1,) )  

# Embedding network
user_embed = Embedding( N+1, K)(user_net)  # dimension is (# of samples, 1, K).  1 is length of sentence in NLP.
movie_embed = Embedding( M+1, K)(movie_net)

# Flatten the shape to make it (# of samples, K), then combine.
user_embed = Flatten()(user_embed)
movie_embed = Flatten()(movie_embed)
x = Concatenate()( [ user_embed, movie_embed ] )  #Concatenate means to simply join two arrays. dimension = (# of samples, 2*K)

# Feed forward net.
x = Dense( 512, activation='relu' )(x)
output_net = Dense(1)(x)

model = Model( inputs=[user_net, movie_net], outputs=output_net )

#---------------------------------------------------------------------
# Define loss, optimizer, and compile it.

my_loss = tf.keras.losses.MeanSquaredError( )  
my_optimizer = tf.keras.optimizers.RMSprop( learning_rate=0.001 )
model.compile( loss=my_loss, optimizer=my_optimizer )

#---------------------------------------------------------------------
# Prepare the data.

user_ids, movie_ids, ratings = shuffle(user_ids, movie_ids, ratings)
Ntrain = int(0.8 * len(ratings))
train_user = user_ids[:Ntrain]
train_movie = movie_ids[:Ntrain]
train_ratings = ratings[:Ntrain]

test_user = user_ids[Ntrain:]
test_movie = movie_ids[Ntrain:]
test_ratings = ratings[Ntrain:]

# Normalize the ratings
avg_rating = train_ratings.mean()
train_ratings = train_ratings - avg_rating
test_ratings = test_ratings - avg_rating

#-----------------------------------------------------------------------
# Train the data.

res = model.fit( 
				[train_user, train_movie], 
				train_ratings, 
				epochs=20, 
				batch_size=256, 
				verbose=2,
				validation_data=([test_user, test_movie], test_ratings),
				)
				
# plot losses
plt.plot(res.history['loss'], label="train loss")
plt.plot(res.history['val_loss'], label="val loss")
plt.legend()
plt.show()

# Now predict.  (don't forget to normalize back rating)
user_id_pred = np.array([test_user[32]])
movie_id_pred = np.array( [test_movie[16]] )
print("rating = ", model.predict( [user_id_pred, movie_id_pred] ) + avg_rating	)	

# Find movies in which a user hasn't watched yet.  predict the rate and recommend it.
				
#import pdb; pdb.set_trace()  




