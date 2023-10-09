import itertools

from A3C import A3C
from Network import Network


class Worker:
    def __init__(self, id_name, global_counter, master_network, env_name, epsilon ):
        self.id = id_name
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.master_network = master_network
        self.env_name = env_name
        self.epsilon = epsilon
        
        local_network = Network( master=False, id_name = self.id )
        
        self.agent = A3C( self.id, self.master_network, local_network, self.env_name, self.epsilon )
        
    def run(self, coordinator):  # coordinator=tf.train.Coordinator()
        self.master_network = self.agent.play() 
    
        '''
        for n in range(20):
            time.sleep(np.random.rand()*2)
            global_step = next(self.global_counter)
            local_step = next(self.local_counter)
            print("Worker({}): {}: {}".format(self.id, local_step, global_step))
        '''
