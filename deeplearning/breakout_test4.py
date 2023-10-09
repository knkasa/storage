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
import gc
import random as rnd

#  Episode length =  5 live.  action=2,3 

#================================================================================================================
# Example of ALE/Breakout-v5.   Install gym (open AI gym).
# https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
# Note you need to install "pip install ale-py" and   "pip install gym[accept-rom-license]" (for license purpose)
# Notice epsilon becomes smaller as it runs.  Smaller episode means let agent decide actions more frequently.
# https://gym.openai.com/envs/#classic_control   documentationo
# https://keras.io/examples/rl/deep_q_network_breakout/
# https://github.com/yuishihara/A3C-tensorflow
# https://stats.stackexchange.com/questions/304532/a3c-implementation-using-tensorflow
#================================================================================================================

current_dir = 'C:/my_working_env/deeplearning_practice/'
os.chdir( current_dir ) 

# This will give a list of all environment.
#print(  envs.registry.all()  )   

EPOCHS = 10000  # number of episodes to play the game. 
NUM_WORKERS = 8
epsilon = 0.8   # epsilon-greedy.  1.0 means always random.  
learning_rate = 0.0001


class A3C:
    def __init__(self, id_name, master_model, env, epsilon, im_size, m, action_size, batch_size=64):
    
        self.id = id_name
        self.master_model = master_model
        self.optimizer = tf.keras.optimizers.Adam( lr=learning_rate )
        
        self.env_name = env
        self.action_size = action_size       # action_size = 2 (left, right)
        #self.memory = deque( maxlen=100000 )     # deque used to append, pop items in list quickly.  
        self.batch_size = batch_size
        self.im_size = im_size    # truncated image size 84x84x4  (4 past frames)  
        self.m = m     # number of image frames = 4  (3 are past frames)
        
        self.gamma = 1.0      # discount factor in Q function
        self.epsilon = epsilon      # used for epsilon-greedy actions.  1.0 means always take random action.
        
        self.local_network()  # get local network.  
        
        # copy weights from master to local.  
        self.local_model.set_weights( weights=self.master_model.get_weights() )

    def local_network(self):
        self.input_layer = tf.keras.layers.Input( shape=( self.im_size, self.im_size, self.m ), name='input_layer' )  # input dimension 1x84x84x4
        self.hidden_layer1 = tf.keras.layers.Conv2D( 16,  (2,2), activation='relu', strides=2,  name='hidden_layer1' )(self.input_layer) 
        self.hidden_layer2 = tf.keras.layers.Conv2D( 32,  (2,2), activation='relu', strides=2,  name='hidden_layer2' )(self.hidden_layer1) 
        self.hidden_layer3 = tf.keras.layers.Conv2D( 64,  (2,2), activation='relu', strides=2,  name='hidden_layer3' )(self.hidden_layer2) 
        self.flat_layer = tf.keras.layers.Flatten(name='flat_layer')(self.hidden_layer3)
        self.drop_layer1 = tf.keras.layers.Dropout(0.2)(self.flat_layer)
        self.dense_layer = tf.keras.layers.Dense( 128, activation='relu',  name='dense_layer' )(self.flat_layer)
        self.drop_layer2 = tf.keras.layers.Dropout(0.2)(self.dense_layer)
        self.policy_layer = tf.keras.layers.Dense( self.action_size, activation='softmax', name='policy_layer' )(self.drop_layer2)  
        self.value_layer = tf.keras.layers.Dense( 1, name='value_layer' )(self.drop_layer2)
        self.local_model = tf.keras.Model( inputs=self.input_layer, outputs=[self.policy_layer, self.value_layer], name="local_model")

    # create tuple(replay buffer) for (s, a, r, s') 
    #def remember(self, state, action, reward, next_state, done):
    #    self.memory.append( (state, action, reward, next_state, done) )  
    
    def choose_action(self, state, epsilon):   # epsilon-greedy action
        #'''
        if self.id==0:
            #print( self.local_model(state)[0] ) 
            #return   self.fix_action( np.argmax( self.local_model(state)[0] ) )
            return   np.random.choice( [2,3] )
        else:
            return  np.random.choice( [2,3] )
        #'''
        
        '''
        # epsilon-greedy
        if np.random.random() <= epsilon:
            return  self.env.action_space.sample()   # take random actions
        else:
            return  np.argmax( self.local_model(state)[0] )
        '''

    def fix_action(self, action0):
        if action0 == 0:
            return 2   # move left
        elif action0 == 1:
            return 3   # move right

    def preprocess_state(self, img):
        img_temp = img[31:195]  # truncate unnecessary part of the image. 
        img_temp = tf.image.rgb_to_grayscale(img_temp)
        img_temp = tf.image.resize( img_temp, [self.im_size, self.im_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        img_temp = tf.cast( img_temp, tf.float32 )
        return  img_temp[:,:,0]
        
    # Combine image from the past.  
    def combine_images(self, state, next_state):
        state = np.roll( state, 1, axis=3)
        state[0,:,:,0] = next_state
        return  tf.convert_to_tensor( state, dtype=tf.float32 )  
        
    # Stack 4 images to make it 1x84x84x4 dimension.  
    def initialize_image(self, state):
        im = np.stack( [state]*4, axis=2 )
        return  tf.expand_dims( im, 0)
        
    def train(self, tape, reward_list, policy_list, value_list, action_list, batch_size):    # this is for training.    

        # we pick random replay buffers for training. 
        #minibatch = random.sample( self.memory, min(len(self.memory), batch_size) )
        
        del policy_list[-1]
        del reward_list[-1]
        del action_list[-1]
                
        # Note G(sum of future returns) = r + gamma*G (recursive) = r1 + gamma^2*r2 + gamma^3*r3 + ...
        action_list = tf.one_hot( action_list, self.action_size, dtype=tf.float32 ).numpy().tolist()  # one-hot encode.  4=number of actions.  
        advantage = tf.add( reward_list, tf.squeeze( tf.subtract( value_list[1:], value_list[:-1] ) ) )
        policy_responsible = tf.reduce_sum( tf.squeeze(policy_list)*action_list, axis=1 )
        value_loss = tf.reduce_mean( tf.square(advantage) )
        entropy = -tf.reduce_sum(  policy_list*tf.math.log( tf.clip_by_value( policy_list, 1e-10, 1 ) ) )
        policy_loss = tf.reduce_mean( tf.math.log( policy_responsible + 1e-10 )*tf.stop_gradient(advantage) )
        loss = 0.5*value_loss + policy_loss + 0.01*entropy
        
        grad = tape.gradient(target=loss, sources=self.local_model.trainable_variables, output_gradients=None, unconnected_gradients=tf.UnconnectedGradients.NONE)
        grad_clip, global_norm = tf.clip_by_global_norm(t_list=grad, clip_norm=5.0)
        grad_clip[0] = tf.where( tf.math.is_nan(grad_clip[0]), tf.zeros_like(grad_clip[0]), grad_clip[0] )
        rt = self.optimizer.apply_gradients( zip(grad_clip, self.master_model.trainable_variables) )
        

    def play(self):
        
        score_list = []
        for e in range(EPOCHS):
                       
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.reset()
                tape.watch( self.local_model.get_layer('hidden_layer1').trainable_variables )
                tape.watch( self.local_model.get_layer('hidden_layer2').trainable_variables )
                tape.watch( self.local_model.get_layer('hidden_layer3').trainable_variables )
                tape.watch( self.local_model.get_layer('dense_layer').trainable_variables )
                tape.watch( self.local_model.get_layer('policy_layer').trainable_variables )
                tape.watch( self.local_model.get_layer('value_layer').trainable_variables )
                
                env = gym.make( 'ALE/Breakout-v5' )
                state = env.reset()     # get random initial state.
                state = self.preprocess_state(state)  
                
                # image needs to be 1x84x84x4 dimension.  
                state = self.initialize_image(state)  
            
                done = False
                sum_reward = 0      # cumulative reward. 
                action_list = []
                state_list = [state]
                reward_list = []
                policy_list = []
                value_list = []
                env.step( 1 )  # this will place the ball at beginneing.  
                live = 5
                num_run = 0
                while not done:  # get params for one episode.
                    num_run += 1
                    action = self.choose_action( state, self.epsilon )
                    next_state, reward, done, info = env.step(action)
                    next_state = self.preprocess_state(next_state)
                    state = self.combine_images( state, next_state ) 
                    #self.remember( state, action, reward, next_state, done )
                    
                    next_live = info['lives']
                    if next_live!=live:  
                        live = next_live
                        env.step( 1 )
                    
                    if num_run >= 500:  # sometimes the game is bugged, and it will start without the ball.  
                        done = True

                    sum_reward += reward    # get cumulative reward.  
                    if self.id==0: print( num_run,  "action = ", action, " live = ", live , " reward = ", reward, " done = ", done )

                    action_list.append(action)
                    state_list.append(state)
                    reward_list.append(reward)
                    policy_list.append( self.local_model(state)[0] )
                    value_list.append( self.local_model(state)[1] )
                    gc.collect()
                    
                score_list.append(sum_reward)
                mean_score = np.mean(  score_list[-20:]  )

                if self.id==0:
                    print( " ID = ", self.id, " Episode = ", e,  " Score = ", sum_reward, " num_run = ", num_run )
                    if e%1==0:  print( self.local_model(state)[0] )
                    
                self.train( tape, rnd.sample(reward_list,16), rnd.sample(policy_list,16), rnd.sample(value_list,16), rnd.sample(action_list,16), self.batch_size )  # train one episode.
                self.local_model.set_weights( weights=self.master_model.get_weights() )
                    
                # not sure if this is needed.  
                #tape.reset()
                #tape.watch( self.local_model.get_layer('hidden_layer').trainable_variables )
                #tape.watch( self.local_model.get_layer('policy_layer').trainable_variables )
                #tape.watch( self.local_model.get_layer('value_layer').trainable_variables )
                
                #self.epsilon = 0.99*self.epsilon  # Adjust epsilon every episodes.
                
            tf.keras.backend.clear_session()
            gc.collect()
                
        return self.master_model
        
#-------------------------------------------------------------------------------------------------
        
class Worker:
    def __init__(self, id_name, global_counter, master_model, env, epsilon, im_size, m, action_size ):
        self.id = id_name
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.master_model = master_model
        self.env = env
        self.epsilon = epsilon
        self.im_size = im_size
        self.m = m
        self.action_size = action_size
        
        self.agent = A3C( self.id, self.master_model, self.env, self.epsilon, self.im_size, self.m, self.action_size )
        
    def run(self, coordinator):  # coordinator=tf.train.Coordinator()
        self.master_model = self.agent.play() 
    
        '''
        for n in range(20):
            time.sleep(np.random.rand()*2)
            global_step = next(self.global_counter)
            local_step = next(self.local_counter)
            print("Worker({}): {}: {}".format(self.id, local_step, global_step))
        '''
        
    def fix_action(self, action0):
        if action0 == 0:
            return 2   # move left
        else:
            return 3   # move right
        
#-------------------------------------------------------------------------------------------------

class lancher:
    def __init__(self):

        # create workers
        with tf.device("/cpu:0"):

            env = 'ALE/Breakout-v5'  #gym.make( 'ALE/Breakout-v5' ) 
            self.action_size = 2       # action_size = left or right
            self.im_size = 84   # size of image (84x84) that will be truncated to. 
            self.m = 4    # number of images (current and past images) to combine.  
            
            # get master neural network.  
            self.master_network()

            # learning 
            workers = []
            global_counter = itertools.count()
            for worker_id in range(NUM_WORKERS):
                worker = Worker(worker_id, global_counter, self.master_model, env, epsilon, self.im_size, self.m, self.action_size )
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
        env = gym.make( 'ALE/Breakout-v5' ) 
        for n in range(20):
        
            state = env.reset()
            state = self.preprocess_state(state) 
            
            # just making sure input data size needs past and current image combined.
            state = self.combine_images(state, state)                  
            
            done = False
            sum_reward = 0
            env.step( 1 )  
            live = 5
            while not done:
                action = worker.fix_action( np.argmax( worker.master_model.predict(state)[0] ) )
                print( worker.master_model.predict(state)[0] )
                next_state, reward, done, info = env.step(action)
                next_state = self.preprocess_state(next_state)
                next_state = self.combine_images( next_state, state )  
                state = next_state
                sum_reward += reward
                
                next_live = info['lives']
                if next_live!=live:  
                    env.step( 1 )   
                    live = next_live
                #print( action, live , reward, done )
                
                if done: print( "Episode = ", n+1, " Sum reward = ", sum_reward )
            score_list.append( sum_reward )
        print(" Average score = ", np.mean( score_list ) )

        import pdb; pdb.set_trace()  
        env.close()
        
    def master_network(self):
        self.input_layer = tf.keras.layers.Input( shape=( self.im_size, self.im_size, self.m ), name='input_layer' )  
        self.hidden_layer1 = tf.keras.layers.Conv2D( 16,  (2,2), activation='relu', strides=2,  name='hidden_layer1' )(self.input_layer) 
        self.hidden_layer2 = tf.keras.layers.Conv2D( 32,  (2,2), activation='relu', strides=2,  name='hidden_layer2' )(self.hidden_layer1) 
        self.hidden_layer3 = tf.keras.layers.Conv2D( 64,  (2,2), activation='relu', strides=2,  name='hidden_layer3' )(self.hidden_layer2) 
        self.flat_layer = tf.keras.layers.Flatten(name='flat_layer')(self.hidden_layer3)
        self.drop_layer1 = tf.keras.layers.Dropout(0.2)(self.flat_layer)
        self.dense_layer = tf.keras.layers.Dense( 128, activation='relu',  name='dense_layer' )(self.flat_layer)
        self.drop_layer2 = tf.keras.layers.Dropout(0.2)(self.dense_layer)
        self.policy_layer = tf.keras.layers.Dense( self.action_size, activation='softmax', name='policy_layer' )(self.drop_layer2)  
        self.value_layer = tf.keras.layers.Dense( 1, name='value_layer' )(self.drop_layer2)
        self.master_model = tf.keras.Model( inputs=self.input_layer, outputs=[self.policy_layer, self.value_layer], name="master_model")

    def preprocess_state(self, img):
        img_temp = img[31:195]  # truncate unnecessary part of the image. 
        img_temp = tf.image.rgb_to_grayscale(img_temp)
        img_temp = tf.image.resize( img_temp, [self.im_size, self.im_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        img_temp = tf.cast( img_temp, tf.float32 )
        return  img_temp[:,:,0]
        
    # Combine image from the past.  
    def combine_images(self, img1, img2):
        if len(img1.shape)==3 and img1.shape[0]==self.m:
            im = np.append( img1[1:,:,:], np.expand_dims( img2, 0), axis=2 )
            return  tf.expand_dims( im, 0)
        else:
            im = np.stack( [img1]*self.m, axis=2 )
            return  tf.expand_dims( im, 0)

def main():
    lancher()

if __name__ == "__main__":
    main()




