import tensorflow as tf


class Network:
    def __init__(self, master, id_name=None):
    
        if master:
            network_name = "master_model"
        else:
            network_name = "local_model" + str(id_name)
            
        activation_func = 'tanh'
        num_unit = 100
        input_size = 4
        action_size = 2
    
        # Neural network
        input_layer = tf.keras.layers.Input( shape=(input_size,), name='input_layer' )  
        hidden_layer = tf.keras.layers.Dense( num_unit, activation=activation_func, kernel_initializer=tf.random_uniform_initializer(seed=3), name='hidden_layer' )(input_layer)  
        policy_layer = tf.keras.layers.Dense( action_size, activation='softmax', kernel_initializer=tf.random_uniform_initializer(seed=3), name='policy_layer' )(hidden_layer)  
        value_layer = tf.keras.layers.Dense( 1, kernel_initializer=tf.random_uniform_initializer(seed=3), name='value_layer' )(hidden_layer)  
        self.model = tf.keras.Model( inputs=input_layer, outputs=[policy_layer, value_layer], name=network_name)
