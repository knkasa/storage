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
# Example of .
# Note you need to install "pip install ale-py" and 
#     "pip install gym[accept-rom-license]" (for license purpose)
# https://gym.openai.com/envs/#classic_control   documentation
#=================================================================

current_dir = 'C:/my_working_env/deeplearning_practice/'
os.chdir( current_dir ) 

import pdb; pdb.set_trace()  


# This will output a list of all environment.
print(  envs.registry.all()  )   
env = gym.make( "ALE/Breakout-v5" )
print( env.observation_space.shape[0] )
print( env.action_space.n )   # there 4 actions to take (0=???, 1=put ball in environment, 2=left, 3=right)

# First, reset the environment.
state = env.reset()

# Play game and save each frame as list.
frames = []
done = False
live = 5
env.step( 1 )  
action_list = [ 2,  3]  # 2=left, 3=right 
for n in range(1000):
    img_array = env.render(mode='rgb_array')
    frames.append( img_array )
    #state, reward, done, info = env.step( 2 )   
    state, reward, done, info = env.step( np.random.choice(action_list) ) 
    print( n,  live, reward ) 
    next_live = info['lives']
    
    if next_live!=live:  
        env.step( 1 )   
        live = next_live
    #plt.imshow( img_array )
    #plt.show()  
    #print( live , done )
        
 
# save it as a gif
clip = ImageSequenceClip(list(frames), fps=10)
clip.write_gif('test.gif', )



#import pdb; pdb.set_trace()  
env.close()
