import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Using pre-trained tensorflow-hub, but use transformers Bert for tokenization.

# Sample text data and labels
texts = ["This is a positive review.", "Negative sentiment detected.", "Neutral statement here."]
labels = [1, 0, 2]  # Labels: 1 (positive), 0 (negative), 2 (neutral)

# Load a pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the text data and convert to input IDs
max_sequence_length = 100  # Set your desired sequence length
input_ids = []
attention_masks = []
for text in texts:
    tokenized_text = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_sequence_length, pad_to_max_length=True, return_attention_mask=True, return_tensors='tf')
    input_ids.append(tokenized_text['input_ids'])
    attention_masks.append(tokenized_text['attention_mask'])

input_ids = tf.concat(input_ids, axis=0)
attention_masks = tf.concat(attention_masks, axis=0)

# Convert labels to TensorFlow constant
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Load a pre-trained embedding model from TensorFlow Hub (e.g., Universal Sentence Encoder)
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

# Convert texts to a NumPy array with the appropriate data type
texts = np.array(texts)

model.fit(texts, y, epochs=epochs, batch_size=batch_size)

# Make predictions on new text data
new_text = ["This is a new text to classify.", "Another neutral statement."]
input_ids_new = []
attention_masks_new = []
for text in new_text:
    tokenized_text = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_sequence_length, pad_to_max_length=True, return_attention_mask=True, return_tensors='tf')
    input_ids_new.append(tokenized_text['input_ids'])
    attention_masks_new.append(tokenized_text['attention_mask'])

input_ids_new = tf.concat(input_ids_new, axis=0)
attention_masks_new = tf.concat(attention_masks_new, axis=0)

predictions = model.predict(new_text)
print("Predictions:", predictions)
