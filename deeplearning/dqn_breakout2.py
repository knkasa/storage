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
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

#================================================================================================================
# Example of ALE/Breakout-v5.   Install gym (open AI gym).
# https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
# Note you need to install "pip install ale-py" and   "pip install gym[accept-rom-license]" (for license purpose)
# Notice epsilon becomes smaller as it runs.  Smaller episode means let agent decide actions more frequently.
# Note it will take about 10000 episodes to learn.  
# https://gym.openai.com/envs/#classic_control   documentation

# https://github.com/sudharsan13296/Deep-Reinforcement-Learning-With-Python/blob/master/09.%20%20Deep%20Q%20Network%20and%20its%20Variants/9.03.%20Playing%20Atari%20Games%20using%20DQN.ipynb
# https://towardsdatascience.com/practical-guide-for-dqn-3b70b1d759bf
# https://gist.github.com/czxttkl/9cc879b9881fef3f79d36ed7e12b53a6
# https://github.com/AdrianHsu/breakout-Deep-Q-Network
#================================================================================================================

current_dir = 'C:/my_working_env/deeplearning_practice/'
os.chdir( current_dir ) 

# This will give a list of all environment.
#print(  envs.registry.all()  )   

EPOCHS = 10000  # Number to run episodes. 
THRESHOLD = 1000000  # average score (last 10 trial) to achieve. 

        
class DQN:
    def __init__(self, env_string, batch_size=64, IM_SIZE=84, m=4 ):
        self.memory = deque( maxlen=100000 )  
        self.env = gym.make( env_string )
        input_size = self.env.observation_space.shape[0]  
        action_size = 2  
        self.batch_size = batch_size
        
        self.gamma = 1.0  # Discount factor in Q function
        self.epsilon = 1.0   # For epsilon-greedy
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.IM_SIZE = IM_SIZE  # image size 84x84
        self.m = m    # number of images (current and past images) to combine.  
        alpha = 0.01  # this is for optimizer
        alpha_decay = 0.01  # this s for optimizer
        
        # Set up CNN network.
        self.model = Sequential()
        self.model.add( Conv2D(32, 8, (4,4), activation='relu', padding='same', input_shape=( IM_SIZE, IM_SIZE, m ) ) )
        self.model.add( Conv2D(64, 4, (2,2), activation='relu', padding='valid' ) )
        self.model.add( Conv2D(64, 3, (1,1), activation='relu', padding='valid' ) )
        self.model.add( Flatten() )
        self.model.add( Dense( 256, activation='elu' ) )
        self.model.add( Dense(action_size, activation='linear') )
        self.model.compile( loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay) )

    # Resize to 84x84 pixels, convert to grayscale.  
    def preprocess_state(self, img):
        img_temp = img[31:195]
        img_temp = tf.image.rgb_to_grayscale(img_temp)
        img_temp = tf.image.resize( img_temp, [self.IM_SIZE, self.IM_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
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
            
    # create tuple(replay buffer) for (s, a, r, s') 
    def remember(self, state, action, reward, next_state, done):
        self.memory.append( (state, action, reward, next_state, done) )  

    def fix_action(self, action0):
        if action0 == 0:
            return 2   # move left
        elif action0 == 1:
            return 3   # move right
            
    def revert_action(self, action):
        if action==2:
            return 0
        elif action==3:
            return 1

    def choose_action(self, state, epsilon):   # epsilon-greedy action
        if np.random.random() <= epsilon:
            return  np.random.choice( [2,3] )   # take random actions
        else:
            return  self.fix_action( np.argmax( self.model.predict(state) ) )

    def replay(self, batch_size):    # this is for training.    

        # we pick random replay buffers for training. 
        minibatch = random.sample( self.memory, min(len(self.memory), batch_size) )
        
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in minibatch:
        
            # calculate returns using Q (value-state function)
            # Note G = r + gamma*G (recursive) = r1 + gamma^2*r2 + gamma^3*r3 + ...
            Q_func = reward + self.gamma*np.max( self.model.predict(next_state)[0] )
            y_target = self.model.predict(state)
            if done:
                y_target[0, self.revert_action(action)] = reward
            else:
                y_target[0, self.revert_action(action)] = Q_func

            x_batch.append(state[0])
            y_batch.append(y_target[0])
            print( self.model.predict(state) , y_target[0] ) 
        
        self.model.fit( np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0 )
        #self.model.fit( np.array(x_batch), np.array(y_batch), batch_size=len(x_batch),  )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        scores = []
        avg_scores = []
        
        for e in range(EPOCHS):
            state = self.env.reset()
            self.env.step( 1 )  
            state = self.preprocess_state(state)
            
            # just making sure input data size needs past and current image combined.
            state = self.initialize_image(state)  
            
            done = False
            i = 0
            # If it breaks one block, one point.  
            live = 5
            num_run = 0
            while not done:
                num_run += 1
                action = self.choose_action( state, self.epsilon )
                next_state, reward, done, info = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                next_state = self.combine_images( state, next_state ) 
                self.remember( state, action, reward, next_state, done )    # combine in tuple
                state = next_state
                
                next_live = info['lives']
                if next_live!=live:  
                    self.env.step( 1 )   
                    live = next_live
                    done = True
                if num_run >= 500:
                    done = True
                    
                #print( num_run,  "action = ", action, " live = ", live , " reward = ", reward, " done = ", done )
                
                # Decrease epsilon after each actions.
                self.epsilon = max( self.epsilon_min, self.epsilon_decay*self.epsilon ) 

                i += reward  # Get cumulative reward.  

            scores.append(i)
            mean_score = np.mean( scores[-10:] )
            avg_scores.append(mean_score)
            
            if mean_score >= THRESHOLD:
                print( e, " episodes run.  Finished!! " )
                return  avg_scores
            else:
                print( e, " episodes run.  Score = ", i, ".  mean_score(last 10 trials) = ", 
                        mean_score )

            self.replay( self.batch_size )  # train one episode.

        return avg_scores

agent = DQN( 'ALE/Breakout-v5' )
scores = agent.train() 

import pdb; pdb.set_trace()  
env.close()





