import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data and labels
texts = ["This is a positive review.", "Negative sentiment detected.", "Neutral statement here."]
labels = [1, 0, 2]  # Labels: 1 (positive), 0 (negative), 2 (neutral)

# Preprocess text data
max_sequence_length = 100  # Set your desired sequence length
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert labels to TensorFlow constant
y = tf.constant(labels)

# Load a pre-trained embedding model from TensorFlow Hub (e.g., Universal Sentence Encoder)
# If error happens, try downloading the file to local, and specifiy the directry below.
embedding_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
hub_layer = hub.KerasLayer(embedding_url, input_shape=[], dtype=tf.string, trainable=False)

# Build the text classification model
input_text = Input(shape=(), dtype=tf.string)
embedding = hub_layer(input_text)
dense = Dense(128, activation='relu')(embedding)
output = Dense(3, activation='softmax')(dense)  # 3 output classes (0, 1, 2)

model = Model(inputs=input_text, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 10  # Set the number of training epochs
batch_size = 32  # Set the batch size
model.fit(tf.constant(texts), y, epochs=epochs, batch_size=batch_size)

# Make predictions on new text data
new_text = ["This is a new text to classify.", "Another neutral statement."]
new_sequences = tokenizer.texts_to_sequences(new_text)
new_X = pad_sequences(new_sequences, maxlen=max_sequence_length)
predictions = model.predict(tf.constant(new_text))
print("Predictions:", predictions)
