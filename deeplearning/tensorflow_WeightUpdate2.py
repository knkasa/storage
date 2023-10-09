import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print()

class network_model():
    def __init__(self, num_inputs):
    
        # Define model here
        self.inputs = num_inputs
        self.input_val = tf.keras.Input(shape=(self.inputs,), name='input_layer' )   
        self.layer_val = tf.keras.layers.Dense(10, activation="relu", name='hidden_layer')(self.input_val)   # 1sgt hidden layer  
        self.outputs = tf.keras.layers.Dense(1, name='output_layer')(self.layer_val)   
        self.model = tf.keras.Model(inputs=self.input_val, outputs=self.outputs, name="xxxx_model")
        
        #if there are 10 columns for output 
        #self.outputs = tf.keras.layers.Dense(10, name='output_layer')(self.layer_val)         
        #self.outputs2 = tf.keras.layers.Reshape((5,2))
        #self.outputs3 = tf.keras.layers.Activation(("sigmoid"))
    
    def get_loss(self, input, target):
        # Here, you could also use "output_value = self.model(inut)"
        input_value = self.model.get_layer(name="input_layer", index=None).call(input)
        hidden_value = self.model.get_layer(name="hidden_layer", index=None).call(input_value)
        output_value = self.model.get_layer(name="output_layer", index=None).call(hidden_value)
        loss = tf.reduce_mean(tf.abs(target - tf.squeeze(output_value)  )  )
        #loss = tf.keras.losses.MSE( tf.squeeze(input), tf.squeeze(target)  )    # If you want to use built-in loss func
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=actions)
        return loss

    def get_grad(self, input, target):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.model.get_layer("hidden_layer").trainable_variables)
            tape.watch(self.model.get_layer("output_layer").trainable_variables)
            loss = self.get_loss( input, target)
        xgrad = tape.gradient(
                    target=loss,
                    sources=self.model.trainable_variables,
                    output_gradients=None,
                    unconnected_gradients=tf.UnconnectedGradients.NONE
                )
        return xgrad

        
num_inputs = 784  #number of column in input data

machine = network_model(num_inputs)
print( machine.model.summary()  )

optimizer=tf.keras.optimizers.RMSprop()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # input data in numpy format
x_train = x_train.reshape(60000, 784).astype("float32") / 255   # need to normalize it
x_test = x_test.reshape(10000, 784).astype("float32") / 255

tf_test = tf.convert_to_tensor( y_test,  dtype=tf.float32)   # data might need to be in tensorflow format

for n in range(10):
    grad = machine.get_grad( x_test, tf_test )
    optimizer.apply_gradients(zip(grad, machine.model.trainable_variables))

    new_loss = machine.get_loss( x_test, y_test )
    print( new_loss.numpy() )

