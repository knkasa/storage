import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys, os
from gym import envs
import gym  
#import matplotlib.animation as animation
from moviepy.editor import ImageSequenceClip


#=================================================================
# Example of Cartpole.
# Note you need to install "pip install ale-py" and 
#     "pip install gym[accept-rom-license]" (for license purpose)
# https://gym.openai.com/envs/#classic_control   documentation
#=================================================================

current_dir = 'C:/my_working_env/deeplearning_practice/'
os.chdir( current_dir ) 


# This will output a list of all environment.
print(  envs.registry.all()  )   
env = gym.make( "CartPole-v0" )
print( env.observation_space.shape[0] )
print( env.action_space.n )   # there 4 actions to take (0=???, 1=put ball in environment, 2=left, 3=right)

# First, reset the environment.
state = env.reset()

# Play game and save each frame as list.
frames = []
done = False
env.step( 1 )  
action_list = [0, 1]  # 2=left, 3=right 
for n in range(100):
    img_array = env.render(mode='rgb_array')
    frames.append( img_array )
    #state, reward, done, info = env.step( 2 )   
    state, reward, done, info = env.step( np.random.choice(action_list) ) 
     
    print( n,  reward ) 
 
# save it as a gif
clip = ImageSequenceClip(list(frames), fps=30)
clip.write_gif('test.gif', )


#import pdb; pdb.set_trace()  
env.close()

