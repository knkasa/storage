import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# gradient tape explained.  https://qiita.com/everylittle/items/20bcbac9bc37dac64534   
# # gradient clipped is needed after getting gradient.  https://qiita.com/rindai87/items/4b6f985c0583772a2e21
# gradient clipped explained.  https://stackoverflow.com/questions/44796793/difference-between-tf-clip-by-value-and-tf-clip-by-global-norm-for-rnns-and-how

  
N = 100
D = 1  # number of columns

# create dataset X, Y
X = np.linspace( 0, 10, N )
X = np.reshape( X, [N,D] )   # shape is NxD
w = np.array( [[0.5]] )     # shape is Dx1
b = 1
Y = np.matmul( X, w) + b + np.random.randn(N, 1) * 0.4
#plt.scatter( X, Y )
#plt.show()

X = tf.convert_to_tensor( X, dtype=tf.float32 )
Y = tf.convert_to_tensor( Y, dtype=tf.float32 )

opt = tf.keras.optimizers.SGD(lr=0.1)

class network():
    def __init__(self, num_column):
    
        # Define model here
        self.num_column = num_column
        self.input_val = tf.keras.Input(shape=(self.num_column,), name='input_layer' )   
        self.hidden_val = tf.keras.layers.Dense(100, activation="sigmoid", name='hidden_layer')(self.input_val)
        self.outputs = tf.keras.layers.Dense(1, name='output_layer')(self.hidden_val)    
        self.model = tf.keras.Model(inputs=self.input_val, outputs=self.outputs, name="xxxx_model")
        print( self.model.summary() )
		
        #if there are 10 columns for output 
        #self.outputs = tf.keras.layers.Dense(10, name='output_layer')(self.layer_val)         
        #self.outputs2 = tf.keras.layers.Reshape((5,2))
        #self.outputs3 = tf.keras.layers.Activation(("sigmoid"))
        
    def get_output(self, input):
        # Here, you could also use "output_val = self.model(inut)"
        input_val = self.model.get_layer( name="input_layer", index=None).call(input)
        hidden_val = self.model.get_layer( name="hidden_layer", index=None).call(input_val)
        output_val = self.model.get_layer( name="output_layer", index=None).call(hidden_val)
        return output_val
        
    def get_loss(self, input, target):
        output_val = self.get_output(input)
        loss = tf.reduce_mean( tf.square(target - output_val) )
        # loss = tf.keras.losses.MSE( tf.squeeze(output_val), tf.squeeze(target)  )
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=action_logits, labels=actions)
        #import pdb; pdb.set_trace()
        return loss

    def get_grad(self, input, target):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:  # gradient tape is needed up to loss calculation
            tape.reset()  
            tape.watch(self.model.get_layer("hidden_layer").trainable_variables)
            tape.watch(self.model.get_layer("output_layer").trainable_variables)
            loss = self.get_loss( input, target)
        xgrad = tape.gradient( loss, self.model.trainable_variables  )
        return xgrad
        

network = network(D)  
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
#optimizer = tf.keras.optimizers.SGD()
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# below is for saving weights
temp_dir = "C:/my_working_env/deeplearning_practice/weights"  
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=network.model)
tmp_ckpoint_tf2 = tf.train.CheckpointManager(
    checkpoint=ckpt,
    directory=temp_dir,
    max_to_keep=1,
    keep_checkpoint_every_n_hours=None,
    checkpoint_name='model_tf2.1'  ) 
#ckpt.restore(tmp_ckpoint_tf2.latest_checkpoint)   #for loading weights
ckpt.restore( tf.train.latest_checkpoint(temp_dir) ) 
network.model = ckpt.model                   #update network just in case


for n in range(100):
    grad = network.get_grad( X, Y )
    #grad_clipped, global_norm = tf.clip_by_global_norm(t_list=grad, clip_norm=5.0)  # clip grad values if too big
    optimizer.apply_gradients(zip(grad, network.model.trainable_variables) )
    print( network.get_loss( X, Y ).numpy() )
    rt_save = tmp_ckpoint_tf2.save(checkpoint_number=n )   # save weights.  this will take time.  


print( network.model.layers[1].get_weights()  )
    
Y_test = network.get_output( X)
plt.plot( X, Y, 'x', X, Y_test )
plt.show()
 

