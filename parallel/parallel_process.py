
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

A = np.random.rand(100,100)
A = tf.constant( A )
print( A )

print("Is there a CPU available: ")
print(tf.config.list_physical_devices("CPU"))
print("Is the Tensor on CPU #0:  ")
print(A.device.endswith("CPU:0"))

def time_matmul(X):
    start = time.time()
    for n in range(1000):
        tf.matmul( X, X)
        
    res = time.time()-start
    print("100 loops (sec) = ", res )
    
time_matmul(A)

print("Using CPU")
with tf.device("CPU:0"):
    assert A.device.endswith("CPU:2")
    time_matmul(A)
