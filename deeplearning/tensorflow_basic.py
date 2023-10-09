# Tensorflow basics 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# If tensorflow version is too new, and need to revert to old version
# like TF >2.5, run this command.  If need, train the network again.  
#tf.compat.v1.disable_v2_behavior()


print()

#scalar calculation
a = tf.constant(3.0)
b = tf.constant(4.0)
c = tf.sqrt(a**2 + b**2)
print(f" c = {c}")
print( c, c.numpy(), c.shape, c.dtype )  # shape=()  means constant

# convert numpy to tensor 
x = np.linspace(0, 1, 11)
x = np.reshape( x, [11,1] )
x = np.asarray(x).astype('float32')  # this is needed to avoid errors
x = tf.convert_to_tensor( x, dtype=tf.float32 )
x = tf.cast(x, dtype=tf.float32)  # change tensor type to another type.
x2 = tf.squeeze(x)   # this will change dimension from (11,1) to (11,)  (basically remove dimension 1)
x2 = x*x   # this is just element-wise multiplication   

# matrix multiplication
M = tf.constant( [[3, 2], [1, 1]] )
x = tf.constant(  [1, 2] )
x = tf.reshape( x, [2,1] )
A = tf.matmul(M,x )    #tf.tensordot for dot product

# indexing tensor as layers.
z = tf.constant([[0,1,2,3]])
tf.keras.layers.Lambda( lambda  x: x[:, 2:] )(z)

x[ tf.newaxis, : ]  # make dimension from (11) to (1,11).  
tf.squeeze(

# Look tf.math for any type of mathematical operations on the web.
M2 = tf.math.add( M, 1 )  # you can add two tensors even if dimension not agree/ (M+1) also works.

print( M2.dtype )

# See what kind of device we are using.  If you use Nvidia with Cuda driver, tensorflow will automatically  use GPU.
tf.config.list_physical_devices()

tf.keras.backend.clear_session()  # clear memory (RAM)


#------------ apply gradient -----------------------------------------------
x = tf.Variable([1,2,3,4,5,6.])  # it's treated variable during GradientTape
c = tf.constant(3.)  #  it's treated constant during GradientTape
with tf.GradientTape() as tape:
    tape.watch(x)   # taking partial derivative respect to x
    z = x*c         # note this needs to be inside "tape" 
    loss = tf.reduce_sum( z**2 )
    print("loss = ", loss.numpy() )
grad = tape.gradient( loss, x )
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# we have grad=[grad1, grad2, ...]
# and  x=[ x1, x2, ... ]
# we want [ (grad1,x1), (grad2,x2), ... ]  using zip.  (Note this is just list of tuple)
opt.apply_gradients(  zip([grad], [x]) )  # or use "zip(grad,x)" or "[(grad,x)]"  if dont work
loss = tf.reduce_sum( (x*c)**2 )    # calculate loss again
print( "loss2 = ", loss.numpy() )   # check loss again
#-----------------------------------------------------------------------------