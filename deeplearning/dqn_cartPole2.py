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

EPOCHS = 1000  # number of episodes to play the game. 
THRESHOLD = 100  # average(last 20 trial) survival time to achieve.  (45 in the book) 
MONITOR = True

class DQN:
    def __init__(self, env_string, batch_size=64):
        self.memory = deque( maxlen=100000 )     # deque used to append, pop items in list quickly.  
        self.env = gym.make( env_string )        # initiate the game environment.
        self.input_size = self.env.observation_space.shape[0]       # shape=4 (x, y, v_x, v_y)
        self.action_size = self.env.action_space.n       # number of action=2 (left, right)
        self.batch_size = batch_size
        
        self.gamma = 1.0      # discount factor in Q function
        self.epsilon = 1.0        # used for epsilon-greedy actions.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        alpha=0.01       # this is for optimizer
        alpha_decay=0.01        # this is for optimizer
        
        # Neural network
        self.model = Sequential()
        self.model.add( Dense(24, input_dim=self.input_size, activation='tanh') )      # input layer
        self.model.add( Dense(48, activation='tanh') )       # one hidden layer
        self.model.add( Dense(self.action_size, activation='linear') )       # output layer. 
        self.model.compile( loss='mse', optimizer=Adam( lr=alpha, decay=alpha_decay )  )

    # create tuple(replay buffer) for (s, a, r, s') 
    def remember(self, state, action, reward, next_state, done):
        self.memory.append( (state, action, reward, next_state, done) )  
        
    def replay(self, batch_size):    # this is for training.    

        # we pick random replay buffers for training. 
        minibatch = random.sample( self.memory, min(len(self.memory), batch_size) )
        
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in minibatch:
        
            # calculate returns using Q (value-state function)
            Q_func = reward + self.gamma*np.max( self.model.predict(next_state)[0] )
            y_target = self.model.predict(state)
            if done:
                y_target[0,action] = reward
            else:
                y_target[0,action] = Q_func

            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit( np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0 )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def choose_action(self, state, epsilon):   # epsilon-greedy action
        if np.random.random() <= epsilon:
            return  self.env.action_space.sample()   # take random actions
        else:
            return  np.argmax( self.model.predict(state) )

    def preprocess_state(self, state):
        return np.reshape( state, [1, self.input_size] )
        
    def train(self):
        
        #scores = deque( maxlen=100 )
        scores = []
        avg_scores = []
        
        for e in range(EPOCHS):
            
            state = self.env.reset()  # get random initial state.
            state = self.preprocess_state(state)  # see below. just reshaping.
            
            done = False
            i = 0      # i = time it stayed alive (in seconds)
            while not done:
                action = self.choose_action( state, self.epsilon ) 
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember( state, action, reward, next_state, done )
                state = next_state
                i += 1   # add 1 time step
                
                # decrease epsilon as time goes by.
                self.epsilon = max( self.epsilon_min, self.epsilon_decay*self.epsilon )
                #self.epsilon = max( self.epsilon_min, min(self.epsilon, 1.0-math.log10( (e+1)*self.epsilon_decay) ) )

                
            scores.append(i)
            mean_score = np.mean(  scores[-20:]  )
            avg_scores.append( mean_score )

            if mean_score >= THRESHOLD:
                print( e, " episodes run.  Finished!! " )
                return  avg_scores
            else:
                print( e, " episodes run. Survival time = ", i, ".  mean_score(last 20 trials) = ", 
                        mean_score, " epsilon = ", self.epsilon )
                
            self.replay( self.batch_size )  # train one episode.
            
        return avg_scores
        
#-------------------------------------------------------------------------------------------------
        
agent = DQN( 'CartPole-v0' )
scores = agent.train() 

import pdb; pdb.set_trace()  
env.close()



