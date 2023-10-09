import numpy as np
import tensorflow as tf
from gym import envs
import gym  
import threading


class A3C:
    def __init__(self, id_name, master_network, local_network, env_name, epsilon, batch_size=64):
    
        self.id = id_name
        self.master_network = master_network
        self.local_network = local_network
        learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam( lr=learning_rate )
        
        self.env_name = env_name
        self.input_size = 4  #self.env.observation_space.shape[0]       # shape=4 (x, y, v_x, v_y)
        self.action_size = 2   #self.env.action_space.n       # number of action=2 (left, right)
        #self.memory = deque( maxlen=100000 )     # deque used to append, pop items in list quickly.  
        self.batch_size = batch_size
        
        self.gamma = 1.0      # discount factor in Q function
        self.epsilon = epsilon      # used for epsilon-greedy actions.  1.0 means always take random action.
        
        # copy weights from master to local.  
        self.local_network.model.set_weights( weights=self.master_network.model.get_weights() )

    # create tuple(replay buffer) for (s, a, r, s') 
    #def remember(self, state, action, reward, next_state, done):
    #    self.memory.append( (state, action, reward, next_state, done) )  
        
    def train(self, tape, reward_list, policy_list, value_list, action_list):    # this is for training.    

        # we pick random replay buffers for training. 
        #minibatch = random.sample( self.memory, min(len(self.memory), self.batch_size) )
                            
        #del policy_list[-1]
        #del reward_list[-1]
        #del action_list[-1]
        
        # Note G(sum of future returns) = r + gamma*G (recursive) = r1 + gamma^2*r2 + gamma^3*r3 + ...
        action_list = tf.one_hot( action_list, self.action_size, dtype=tf.float32 ).numpy().tolist()  # one-hot encode
        #advantage = tf.add( reward_list, tf.squeeze( tf.subtract( value_list[1:], value_list[:-1] ) ) )  #"High-dimentional continuous control using generalized advantage estimation." arXiv]1506.02438(2015)
        advantage = self.get_advantage( reward_list, value_list )  # Same Advantage as the Google paper but note one of V is missing.  
        
        policy_responsible = tf.reduce_sum( tf.squeeze(policy_list)*action_list, axis=1 )
        value_loss = tf.reduce_mean( tf.square(advantage) )
        entropy = -tf.reduce_sum(  policy_list*tf.math.log( tf.clip_by_value( policy_list, 1e-10, 1 ) ) )
        policy_loss = tf.reduce_mean( tf.math.log( policy_responsible + 1e-10 )*tf.stop_gradient(advantage) )
        #policy_loss = tf.reduce_mean( tf.math.log( policy_responsible + 1e-10 )*advantage )
        loss = 0.5*value_loss - policy_loss + 0.01*entropy
        #loss = policy_loss + 0.0*entropy
        
        grad = tape.gradient(target=loss, sources=self.local_network.model.trainable_variables, output_gradients=None, unconnected_gradients=tf.UnconnectedGradients.NONE)
        grad_clip, global_norm = tf.clip_by_global_norm(t_list=grad, clip_norm=5.0)
        grad_clip[0] = tf.where( tf.math.is_nan(grad_clip[0]), tf.zeros_like(grad_clip[0]), grad_clip[0] )
        rt = self.optimizer.apply_gradients( zip(grad_clip, self.master_network.model.trainable_variables) )
        
    def get_advantage(self, reward_list, value_list, gamma=0.99, standardize=True):

        reward_list = tf.concat(  reward_list , axis=0)
        n = tf.shape(reward_list)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums into the `returns` array
        rewards = tf.cast( reward_list[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            discounted_sum = rewards[i] + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]
        
        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns))/(tf.math.reduce_std(returns) + 1.0e-20))
        
        advantage = returns - tf.concat( value_list, axis=0)   # G - V
        
        return advantage

    def choose_action(self, state, epsilon):   # epsilon-greedy action
        return  np.random.choice( self.action_size, p=self.local_network.model(state)[0][0].numpy() )   # action chosen by policy probability
        if np.random.random() <= epsilon:
            return   np.random.choice( [0,1] )  #self.env.action_space.sample()   # take random actions
        else:
            return  np.argmax( self.local_network.model(state)[0] )

    def preprocess_state(self, state):
        return np.reshape( state, [1, self.input_size] )
        
    def play(self):
        
        score_list = []
        EPOCHS = 1000
        for e in range(EPOCHS):
            
            env = gym.make( 'CartPole-v0' )   
            state = env.reset()  # get random initial state.
            state = self.preprocess_state(state)  # see below. just reshaping.
            
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.reset()
                tape.watch( self.local_network.model.get_layer('hidden_layer').trainable_variables )
                tape.watch( self.local_network.model.get_layer('policy_layer').trainable_variables )
                tape.watch( self.local_network.model.get_layer('value_layer').trainable_variables )
            
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
                    reward_list.append(reward*1.0e-2)
                    policy_list.append( self.local_network.model(state)[0] )
                    value_list.append( self.local_network.model(state)[1] )
                    
                score_list.append(i)
                mean_score = np.mean(  score_list[-20:]  )

                if self.id==0:
                    print( " ID = ", self.id, " Episode = ", e,  " Survival time = ", i,
                            ".  mean_score(last 20 trials) = ", np.round(mean_score,2), " epsilon = ", self.epsilon )
                    
                #for n in range(10):
                self.train( tape, reward_list, policy_list, value_list, action_list )  # train one episode.
                self.local_network.model.set_weights( weights=self.master_network.model.get_weights() )
                    
                # not sure if this is needed.  
                #tape.reset()
                #tape.watch( self.local_network.model.get_layer('hidden_layer').trainable_variables )
                #tape.watch( self.local_network.model.get_layer('policy_layer').trainable_variables )
                #tape.watch( self.local_network.model.get_layer('value_layer').trainable_variables )
                
                #self.epsilon = 0.99*self.epsilon  # Adjust epsilon every episodes.
                
        return self.master_network
