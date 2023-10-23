import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import AdamWeightDecay

# Bert multi-label classification model.
# You could also use "flair" library which is built based on BERT for semantic/polarity analysis.

# Define your texts and binary labels
texts = ["Text 1 here.", "Text 2 here.", "Text 3 here."]  # Add your texts here
labels = [0, 1, 2]  # Binary labels, 0 or 1 for each text
n_labels = 3

# Define the pre-trained model and tokenizer (BERT in this case)
#model_name = "bert-base-uncased"
model_name = 'cl-tohoku/bert-base-japanese-v2'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)

# Tokenize and prepare the input data
input_ids = []
attention_masks = []

for text in texts:
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='tf')
    input_ids.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])

input_ids = tf.concat(input_ids, axis=0)
attention_masks = tf.concat(attention_masks, axis=0)
labels = tf.constant(labels)

# Set the BERT layers as trainable if you want.  
for layer in model.layers:
    layer.trainable = True

# Neural network.
input_layer = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
bert_output = model(input_layer)
logits = bert_output.logits
hidden_output = tf.keras.layers.Dense(64, activation='relu')(logits)
classifier_output = tf.keras.layers.Dense(n_labels, activation='softmax')(hidden_output)

classifier = tf.keras.Model(inputs=input_layer, outputs=classifier_output)

# Define optimizer and loss function
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# Compile the model
classifier.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Train the model
classifier.fit(input_ids, labels, epochs=3, batch_size=32)  # You can adjust the number of epochs and batch size




