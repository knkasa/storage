import multiprocessing
import itertools
import tensorflow as tf
import threading
import time
import numpy as np

class Worker:
    def __init__(self, id, global_counter):
        self.id = id
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        #self.network = AC_network()    # define neural network.  

    def work(self, coord):
        while not coord.should_stop():
            time.sleep(1)
            global_step = next(self.global_counter)
            local_step = next(self.local_counter)
            print("Worker({}),  Local_step({}),  Global_step({})".format(self.id, local_step, global_step) )
            if global_step >= 20:
                coord.request_stop()   # this will exit all thread

                
                
NUM_WORKERS = multiprocessing.cpu_count()
NUM_WORKERS = 4  # ID will be 0, 1, 2, 3
global_counter = itertools.count()
coord = tf.train.Coordinator()
print( NUM_WORKERS, global_counter,  coord )

with tf.device("/cpu:0"):  # run the commands below using cpu=0

    # Define the workers with the class "Worker"
    workers = []
    for worker_id in range(NUM_WORKERS):
        worker = Worker(worker_id, global_counter)
        workers.append(worker)

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate thread.
    worker_threads = []
    for w in workers:
        worker_fn = lambda: w.work(coord)  # define class function
        t = threading.Thread(target=worker_fn)  # assign function to each thread
        t.start()    # finally run the function
        worker_threads.append(t)

    print( t )
    #coord.join(worker_threads)  # wait for all thread to finish

    # This is another way to wait all thread.  Wait for all thread to finish.
    for t in worker_threads:
        t.join()

    print( t)   
    
    
