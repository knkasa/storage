import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
from tensorflow import feature_column 
from sklearn.model_selection import train_test_split
import sklearn as sk

#=============================================================
# Example of spam detector.  
# Tokenizer = assign ID to each vocaburary.
# Embedding = convert each word into a vector.
#=============================================================

os.chdir('C:/my_working_env/deeplearning_practice')

#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
#from tensorflow.keras.layers import LSTM, Embedding
#from tensorflow.keras.models import Model

#---------------------------------------------------------------------------------------------------------
# Embedding explained.

sentences = [
    "I like eggs and ham.",
    "I love chocolate and bunnies.",
    "I hate onions.", 
    "I like an egg.",
	]

#------ Instead of Tokenizer, we could use TextVectorization which is a new feature ---------------------
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
tovec = TextVectorization( max_tokens=20000,  # Max number of vocab to set, None also works(no limit).  
                        standardize="lower_and_strip_punctuation",
                        split="whitespace", # delimeter
                        ngrams=None, # ngrams means combine words of n group. "I am" "I was" ...
                        output_mode="int",  # assign integer number to each vocab
                        output_sequence_length=None,  # set number of vocabs for each sentence. None means max length.
                        pad_to_max_tokens=True)  # if true, make all sentence equal length. Does not apply if output_sequence_length=None.
                        
tovec.adapt(sentences)  # adapt is same as fit_on_texts()
print( tovec( [ "I like eggs and ham.",] ) )
exit()

unique_vocab = tovec.get_vocabulary()   
top_5_words = unique_vocab[:5]  # get top 5 common words
least_5_words = unique_vocab[-5:0]  # get top 5 common words
#---------------------------------------------------------------------------------------------------------

MAX_VOCAB_SIZE = 20000  # This is probably good enough. 
tokenizer = tf.keras.preprocessing.text.Tokenizer( num_words=MAX_VOCAB_SIZE )

tokenizer.fit_on_texts(sentences)  # Assign ID number (integer) to each vocaburary.
sequences = tokenizer.texts_to_sequences(sentences)
print( sequences )

# See list of vacab and id 
print( tokenizer.word_index )

# "pad" means to make every sentences an equal lengh of list
data = tf.keras.preprocessing.sequence.pad_sequences(sequences)
print(data)

# We can set the length for padding.  Put zeros at the end.
MAX_SEQUENCE_LENGTH = 10
data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print(data)

# Put zeros at beginning.  Put zeros is better for LSTM.   
data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
print(data)

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# Now for the model.  Create data set.

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Drop uncecessary columns, and rename columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.columns = ['labels', 'data']

# Create binary labels.
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].values

# Split data for training and testing
df_train, df_test, Ytrain, Ytest = sk.model_selection.train_test_split( df['data'], Y, test_size=0.33 )

#-----------------------------------------------------------------------------------------------------
# Prepare the data set.  

# Convert sentences to ID numbers.
MAX_VOCAB_SIZE = 20000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train)
sequences_train = tokenizer.texts_to_sequences(df_train)
sequences_test = tokenizer.texts_to_sequences(df_test)

# Get word -> integer mapping. V is the number to show how many kinds of vocabularies. 
word2idx = tokenizer.word_index
V = len(word2idx)

# Pad sequences to N x T dimension.  T=maximum sentence length.  
data_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_train)
print('Shape of data train tensor:', data_train.shape)

# Get sequence length.  
T = data_train.shape[1]

# Apply pad to test_data with equal length as train_data (length=T).  
data_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, maxlen=T)
print('Shape of data test tensor:', data_test.shape)

#-----------------------------------------------------------------------------------------------
# Create the model.

# We get to choose embedding dimensionality.  This is like number of neurons.  
D = 20

# V is the number to show how many kinds of vocabularies.   
# Note: we actually want to the size of the embedding to (V + 1) x D, (actually, it can be V+1 can be anything as long as it's greater than V+1)
# because the first index starts from 1 and not 0.
# Thus, if the final index of the embedding matrix is V,
# then it actually must have size V + 1.

# Create network.  
# Embedding layer will convert each vocabularies into vector of size D.  
# If two vocaburaries are similar words, then the two vectors will be similar (will point close to each other).  
input_net = tf.keras.layers.Input( shape=(T,) )
x = tf.keras.layers.Embedding(V + 1, D)(input_net)   # Embedding( # of unique vocaburary, # of output columns)
x = tf.keras.layers.LSTM( 15, return_sequences=True )(x)   # activation="tanh" is better?
x = tf.keras.layers.GlobalMaxPooling1D()(x)    # prevents overfitting.  
output_net = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model( inputs=input_net, outputs=output_net )


#--------------------------------------------------------------------------
max_vocab_length = 10000 # max number of words to have in our vocabulary
max_length = 15 # max length our sequences will be (e.g. how many words from a Tweet does our model see?) 
embedding = tf.keras.layers.Embedding(input_dim=max_vocab_length, # set input shape
                             output_dim=128, # set size of embedding vector. You can choose anything.
                             embeddings_initializer="uniform", # default, intialize weights randomly
                             input_length=max_length, # how long is each input
                             name="embedding_1")
#--------------------------------------------------------------------------


# Define optimizer, loss, metrics.  
my_optimizer = tf.keras.optimizers.RMSprop( learning_rate=0.001 )
my_loss = tf.keras.losses.BinaryCrossentropy( from_logits=False )  # False seems better predicted.  
my_metrics = tf.keras.metrics.BinaryAccuracy( threshold=0.5 )   

# Compile the model.  
model.compile( loss=my_loss, optimizer=my_optimizer, metrics=[my_metrics] )
model.__dict__.keys()

res = model.fit( data_train, Ytrain, epochs=5, validation_data=(data_test, Ytest) )

# Plot loss per iteration
plt.plot(res.history['loss'], label='loss')
plt.plot(res.history['val_loss'], label='val_loss')
plt.legend()
plt.show()  

# Plot accuracy per iteration
plt.plot(res.history['binary_accuracy'], label='test_accuracy')
plt.plot(res.history['val_binary_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# Now testing.
pred0 = model.predict( np.expand_dims( data_test[0], axis=0) )  # Remember shape needs to be NxT
print(); print( df_train[0] )
print("prediction = ", np.round(pred0)[0], "actual = ", Ytest[0] )
#import pdb;  pdb.set_trace()  


