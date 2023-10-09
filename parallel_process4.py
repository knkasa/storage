import itertools
import threading
import time
import multiprocessing
import numpy as np
import datetime as dt
from operator import itemgetter
#import tensorflow as tf

class Worker:
    def __init__(self, id, global_counter, dic, num_workers):
        self.id = id
        self.global_counter = global_counter    
        self.local_counter = itertools.count()  # count in local process
        self.dic = dic 
        self.num_workers = num_workers 
    def run(self): 
        xlist = dic[self.id]
        for x in xlist:
            print( "ID = ", self.id, " excuting ", x )
            time.sleep(2)
            
            
#with tf.device("/cpu:0"):
global_counter = itertools.count()
num_workers = multiprocessing.cpu_count()
num_workers = 4  # Process ID will be 0, 1, 2, 3 (for num_workers=4) 
print("Number of workers = ", num_workers )

# list of jobs to execute
xlist = []
for n in range(10):
    x = 'job' + str(n)
    xlist.append(x)
dic = {}
for n in range(num_workers):
    index = np.arange( n, len(xlist), num_workers )
    dic[str(n)] = [ xlist[n]  for n in index ]
    
# create the workers
workers = []
for worker_id in range(num_workers):
    worker = Worker(str(worker_id), global_counter, dic, num_workers)
    workers.append(worker)
    
# start multithread
worker_threads = []
for worker in workers:
    worker_fn = lambda: worker.run()
    t = threading.Thread(target=worker_fn)
    t.start()
    worker_threads.append(t)
    
# end multiprocess
for t in worker_threads:
    t.join()
print("DONE!")



