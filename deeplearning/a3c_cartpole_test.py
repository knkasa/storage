import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys, os
from gym import envs
import gym  
#import matplotlib.animation as animation
from moviepy.editor import ImageSequenceClip

import random
import math
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor as PoolExecutor

#================================================================================================================
# Example of CartPole-v0.   Install gym (open AI gym).
# https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
# Note you need to install "pip install ale-py" and   "pip install gym[accept-rom-license]" (for license purpose)
# Notice epsilon becomes smaller as it runs.  Smaller episode means let agent decide actions more frequently.
# https://gym.openai.com/envs/#classic_control   documentation
#================================================================================================================

current_dir = 'C:/my_working_env/deeplearning_practice/'
os.chdir( current_dir ) 

# This will give a list of all environment.
#print(  envs.registry.all()  )   

EPOCHS = 200 # number of episodes to play the game. 
NUM_WORKERS = 16
epsilon = 1   # epsilon-greedy.  1.0 means always random.  
learning_rate = 0.0001


class A3C:
    def __init__(self, id_name, master_model, env_name, epsilon, batch_size=64):
    
        self.id = id_name
        self.master_model = master_model
        self.optimizer = tf.keras.optimizers.Adam( lr=learning_rate )
        
        self.env_name = env_name
        self.input_size = 4  #self.env.observation_space.shape[0]       # shape=4 (x, y, v_x, v_y)
        self.action_size = 2   #self.env.action_space.n       # number of action=2 (left, right)
        #self.memory = deque( maxlen=100000 )     # deque used to append, pop items in list quickly.  
        self.batch_size = batch_size
        
        self.gamma = 1.0      # discount factor in Q function
        self.epsilon = epsilon      # used for epsilon-greedy actions.  1.0 means always take random action.
        
        # local network
        self.input_layer = tf.keras.layers.Input( shape=(self.input_size,), name='input_layer' )   
        self.hidden_layer = tf.keras.layers.Dense( 48, activation='tanh', name='hidden_layer' )(self.input_layer)  
        self.policy_layer = tf.keras.layers.Dense( self.action_size, activation='softmax', name='policy_layer' )(self.hidden_layer)  
        self.value_layer = tf.keras.layers.Dense( 1,  name='value_layer', activation='linear' )(self.hidden_layer)  
        self.local_model = tf.keras.Model( inputs=self.input_layer, outputs=[self.policy_layer, self.value_layer], name="local_model")
        
        # copy weights from master to local.  
        self.local_model.set_weights( weights=self.master_model.get_weights() )

    # create tuple(replay buffer) for (s, a, r, s') 
    #def remember(self, state, action, reward, next_state, done):
    #    self.memory.append( (state, action, reward, next_state, done) )  
        
    def train(self, tape, reward_list, policy_list, value_list, action_list, batch_size):    # this is for training.    

        # we pick random replay buffers for training. 
        #minibatch = random.sample( self.memory, min(len(self.memory), batch_size) )
                            
        del policy_list[-1]
        del reward_list[-1]
        del action_list[-1]
                
        # Note G(sum of future returns) = r + gamma*G (recursive) = r1 + gamma^2*r2 + gamma^3*r3 + ...
        action_list = tf.one_hot( action_list, self.action_size, dtype=tf.float32 ).numpy().tolist()  # one-hot encode
        advantage = tf.add( reward_list, tf.squeeze( tf.subtract( value_list[1:], value_list[:-1] ) ) )
        policy_responsible = tf.reduce_sum( tf.squeeze(policy_list)*action_list, axis=1 )
        value_loss = tf.reduce_mean( tf.square(advantage) )
        entropy = -tf.reduce_sum(  policy_list*tf.math.log( tf.clip_by_value( policy_list, 1e-10, 1 ) ) )
        policy_loss = tf.reduce_mean( tf.math.log( policy_responsible + 1e-10 )*tf.stop_gradient(advantage) )
        loss = 0.5*value_loss + policy_loss + 0.01*entropy
        #loss = 1.0*value_loss + 1.0e-10*policy_loss
        
        grad = tape.gradient(target=loss, sources=self.local_model.trainable_variables, output_gradients=None, unconnected_gradients=tf.UnconnectedGradients.NONE)
        grad_clip, global_norm = tf.clip_by_global_norm(t_list=grad, clip_norm=5.0)
        grad_clip[0] = tf.where( tf.math.is_nan(grad_clip[0]), tf.zeros_like(grad_clip[0]), grad_clip[0] )
        rt = self.optimizer.apply_gradients( zip(grad_clip, self.master_model.trainable_variables) )
        

    def choose_action(self, state, epsilon):   # epsilon-greedy action
        if np.random.random() <= epsilon:
            return   np.random.choice( [0,1] )  #self.env.action_space.sample()   # take random actions
        else:
            return  np.argmax( self.local_model(state)[0] )

    def preprocess_state(self, state):
        return np.reshape( state, [1, self.input_size] )
        
    def play(self):
        
        score_list = []
        for e in range(EPOCHS):
            
            env = gym.make( 'CartPole-v0' )   
            state = env.reset()  # get random initial state.
            state = self.preprocess_state(state)  # see below. just reshaping.
            
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.reset()
                tape.watch( self.local_model.get_layer('hidden_layer').trainable_variables )
                tape.watch( self.local_model.get_layer('policy_layer').trainable_variables )
                tape.watch( self.local_model.get_layer('value_layer').trainable_variables )
            
                done = False
                i = 0      # i = time it stayed alive (in seconds)
                action_list = []
                state_list = [state]
                reward_list = []
                policy_list = []
                value_list = []
                while not done:  # get params for one episode.  
                    action = self.choose_action( state, self.epsilon ) 
                    next_state, reward, done, _ = env.step(action)
                    next_state = self.preprocess_state(next_state)
                    #self.remember( state, action, reward, next_state, done )
                    state = next_state
                    i += 1     # add 1 time step
                    
                    action_list.append(action)
                    state_list.append(state)
                    reward_list.append(reward)
                    policy_list.append( self.local_model(state)[0] )
                    value_list.append( self.local_model(state)[1] )
                    
                score_list.append(i)
                mean_score = np.mean(  score_list[-20:]  )

                if self.id==0:
                    print( " ID = ", self.id, " Episode = ", e,  " Survival time = ", i,
                            ".  mean_score(last 20 trials) = ", np.round(mean_score,2), " epsilon = ", self.epsilon )
                    
                self.train( tape, reward_list, policy_list, value_list, action_list, self.batch_size )  # train one episode.
                self.local_model.set_weights( weights=self.master_model.get_weights() )
                    
                # not sure if this is needed.  
                #tape.reset()
                #tape.watch( self.local_model.get_layer('hidden_layer').trainable_variables )
                #tape.watch( self.local_model.get_layer('policy_layer').trainable_variables )
                #tape.watch( self.local_model.get_layer('value_layer').trainable_variables )
                
                #self.epsilon = 0.99*self.epsilon  # Adjust epsilon every episodes.
                
        return self.master_model
        
#-------------------------------------------------------------------------------------------------
        
class Worker:
    def __init__(self, id_name, global_counter, master_model, env_name, epsilon ):
        self.id = id_name
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.master_model = master_model
        self.env_name = env_name
        self.epsilon = epsilon
        
        self.agent = A3C( self.id, self.master_model, self.env_name, self.epsilon )
        
    def run(self, coordinator):  # coordinator=tf.train.Coordinator()
        self.master_model = self.agent.play() 
    
        '''
        for n in range(20):
            time.sleep(np.random.rand()*2)
            global_step = next(self.global_counter)
            local_step = next(self.local_counter)
            print("Worker({}): {}: {}".format(self.id, local_step, global_step))
        '''
        
#-------------------------------------------------------------------------------------------------

class lancher:
    def __init__(self):

        # create workers
        with tf.device("/cpu:0"):

            env_name = 'CartPole-v0'  # gym.make( 'CartPole-v0' ) 
            input_size = 4   #env.observation_space.shape[0]       # shape=4 (x, y, v_x, v_y)
            action_size = 2   #env.action_space.n       # number of action=2 (left, right)
            
            # Neural network
            input_layer = tf.keras.layers.Input( shape=(input_size,), name='input_layer' )  
            hidden_layer = tf.keras.layers.Dense( 48, activation='tanh',kernel_initializer=tf.random_uniform_initializer(seed=3), name='hidden_layer1' )(input_layer)  
            policy_layer = tf.keras.layers.Dense( action_size, activation='softmax', kernel_initializer=tf.random_uniform_initializer(seed=3), name='policy_layer' )(hidden_layer)  
            value_layer = tf.keras.layers.Dense( 1, kernel_initializer=tf.random_uniform_initializer(seed=3), name='value_layer' )(hidden_layer)  
            master_model = tf.keras.Model( inputs=input_layer, outputs=[policy_layer, value_layer], name="master_model")

            # learning 
            workers = []
            global_counter = itertools.count()
            for worker_id in range(NUM_WORKERS):
                worker = Worker(worker_id, global_counter, master_model, env_name, epsilon )
                workers.append(worker)

            # start multithread
            worker_threads = []
            cord = tf.train.Coordinator()
            for worker in workers:
                worker_fn = lambda: worker.run( cord )
                t = threading.Thread(target=worker_fn)
                t.start()
                worker_threads.append(t)
            
            # end multithread
            cord.join(worker_threads)
            print(" Done!! ")

        '''
        worker_threads = []
        with PoolExecutor(max_workers=NUM_WORKERS) as executor:
            for worker in workers:
                job = lambda: worker.run(  tf.train.Coordinator() )
                worker_threads.append( executor.submit(job) )
        '''

        # Now for the testing.
        score_list = []
        env = gym.make( 'CartPole-v0' )
        for n in range(30):
            state = env.reset()
            done = False
            sum_reward = 0
            while not done:
                env.render()   # This is to display the game.  
                state = np.reshape( state, [1, env.observation_space.shape[0] ] )  # needs to reshape 
                action = np.argmax( worker.master_model.predict(state)[0] )
                next_state, reward, done, _ = env.step(action)
                state = next_state
                sum_reward += reward
                if done: print( "Episode = ", n+1, " Sum reward = ", sum_reward )
                #print(  "value is ", worker.master_model.predict(np.reshape( state, [1, env.observation_space.shape[0] ] ) )[1], " action is ", worker.master_model.predict(np.reshape( state, [1, env.observation_space.shape[0] ] ) )[0]    )
            score_list.append( sum_reward )
        print(" Average score = ", np.mean( score_list ) )

        import pdb; pdb.set_trace()  
        env.close()

def main():
    lancher()

if __name__ == "__main__":
    main()




