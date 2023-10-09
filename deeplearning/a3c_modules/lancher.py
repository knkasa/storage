import matplotlib.pyplot as plt
from gym import envs
import gym  
import threading
import itertools
import tensorflow as tf

from Worker import Worker
from Network import Network


class lancher:
    def __init__(self):

        # create workers
        with tf.device("/cpu:0"):

            env_name = 'CartPole-v0'  # gym.make( 'CartPole-v0' ) 
            input_size = 4   #env.observation_space.shape[0]       # shape=4 (x, y, v_x, v_y)
            action_size = 2   #env.action_space.n       # number of action=2 (left, right)
            
            # Set user parameters.
            epsilon = 0.5
            NUM_WORKERS = 8
            
            master_network = Network( master=True)

            # learning 
            workers = []
            global_counter = itertools.count()
            for worker_id in range(NUM_WORKERS):
                worker = Worker(worker_id, global_counter, master_network, env_name, epsilon )
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

        # Now for testing.
        score_list = []
        env = gym.make( 'CartPole-v0' )
        frames = []
        for n in range(30):
            state = env.reset()
            done = False
            sum_reward = 0
            while not done:
                env.render()   # This is to display the game.  
                frames.append( env.render(mode='rgb_array') )
                state = np.reshape( state, [1, env.observation_space.shape[0] ] )  # needs to reshape 
                action = np.argmax( worker.master_network.model.predict(state, verbose=0)[0] )
                next_state, reward, done, _ = env.step(action)
                state = next_state
                sum_reward += reward
                if done: print( "Episode = ", n+1, " Sum reward = ", sum_reward )
                #print(  "value is ", worker.master_network.model.predict(np.reshape( state, [1, env.observation_space.shape[0] ] ) )[1], " action is ", worker.master_network.model.predict(np.reshape( state, [1, env.observation_space.shape[0] ] ) )[0]    )
            score_list.append( sum_reward )
        print(" Average score = ", np.mean( score_list ) )
        
        # save it as a gif
        clip = ImageSequenceClip(list(frames), fps=30)
        clip.write_gif('cartpole_minus_housaku.gif', )

        import pdb; pdb.set_trace()  
        env.close()
