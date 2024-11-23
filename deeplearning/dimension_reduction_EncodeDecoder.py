from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

# Define the size of the input and the bottleneck layer (compressed representation)
input_dim = 100  # Example input feature size
encoding_dim = 32  # Bottleneck feature size

# Define the input layer
input_layer = Input(shape=(input_dim,))

# Define the encoder layers
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(encoded)

# Define the decoder layers
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Build the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_val, X_val))

# Extract the encoder model to output features
encoder = Model(inputs=input_layer, outputs=encoded)

# Use the encoder to extract features
X_features = encoder.predict(X_data)  # X_data can be new data for which you want features
