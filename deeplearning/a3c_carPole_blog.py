import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys, os
from gym import envs
import gym, time, random
from datetime import datetime
import gnwrapper
from concurrent.futures import ThreadPoolExecutor as PoolExecutor

#import matplotlib.animation as animation
from moviepy.editor import ImageSequenceClip

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


#=================================================================
# Example of .
# Note you need to install "pip install ale-py" and 
#     "pip install gym[accept-rom-license]" (for license purpose)
# https://gym.openai.com/envs/#classic_control   documentation
#=================================================================

current_dir = 'C:/my_working_env/deeplearning_practice/'
os.chdir( current_dir ) 

'''
# This will output a list of all environment.
print(  envs.registry.all()  )   
env = gym.make( 'CartPole-v0' )
print( env.observation_space.shape[0] )
print( env.action_space.n )   # there 4 actions to take (0=???, 1=put ball in environment, 2=left, 3=right)

score_list = []  
for n in range(10):
    state = env.reset()
    done = False
    sum_reward = 0
    while not done:
        env.render()   # This is to display the game.  
        state = np.reshape( state, [1, env.observation_space.shape[0] ] )     # needs to reshape 
        action = np.random.choice( [0,1] )
        next_state, reward, done, _ = env.step(action)
        state = next_state
        sum_reward += reward
        if done: print( "Episode = ", n+1, " Sum reward = ", sum_reward )
    score_list.append( sum_reward )
print(" Average score = ", np.mean( score_list ) )
'''

#---------------------------------------------------------------------------------------------------------------


# A3C sample: cartpole 
# 以下のURLの実装を TensorFlow 2に変更したもの。
# 参考：https://qiita.com/sugulu/items/acbc909dd9b74b043e45


tf.keras.backend.set_floatx('float64')
start_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

print("Tensorflow version " + tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # TensorFlow高速化用のワーニングを表示させない

# -- constants of Game
ENV = 'CartPole-v0'
env = gym.make(ENV)
NUM_STATES = env.observation_space.shape[0]     # Number of states (x, y, v_x, v_y)
NUM_ACTIONS = env.action_space.n        # 
NONE_STATE = np.zeros(NUM_STATES)

# -- constants of LocalBrain
MIN_BATCH = 5
LEARNING_RATE = 5e-3
RMSPropDecaly = 0.99

# Threashold score for ending the game.  
GOAL = 200

# -- params of Advantage-ベルマン方程式
GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

N_WORKERS = 2   # スレッドの数
Tmax = 10   # 各スレッドの更新ステップ間隔

# ε-greedyのパラメータ
EPS_START = 0.5
EPS_END = 0.0
EPS_STEPS = GOAL*N_WORKERS


# Set up neural network.  
class Network(Model):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden_layer = Dense(32, activation='relu', autocast=False)
        self.out_actions = Dense(NUM_ACTIONS, activation='softmax')  # policy layer
        self.out_value = Dense(1, activation='linear')   # value layer  

    def call(self, x):
        x = self.hidden_layer(x)
        policy_val = self.out_actions(x)
        value_val = self.out_value(x)
        return policy_val, value_val


class GlobalBrain:
    def __init__(self):
    
        self.model = Network()  # set up neural network. 
        self.optimizer = tf.keras.optimizers.Adam( lr=LEARNING_RATE )

        self._activate_weight()  # initialize weights/bias (may not need this).  

        self.isLearned = False  # If score>Goal, it becomes True.  
        self.frames = 0  

    def _activate_weight(self):
        x = np.zeros((1, NUM_STATES))
        predictions = self.model(x)

    # Update gradients.  
    def update_global_weight_params(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    # Update local weight/bias from global worker.
    def pull_global_weight_params(self, local_model):
        [local_weights.assign(global_weights) for local_weights, global_weights in zip(local_model.trainable_weights, self.model.trainable_weights)]

    # Update global weight/bias from local worker.
    def push_local_weight_params(self, local_model):
        [global_weights.assign(local_weights) for global_weights, local_weights in zip(self.model.trainable_weights, local_model.trainable_weights)]


class LocalBrain:
    def __init__(self, id, global_brain):   # globalなglobal_brainを変数として持つ
        with tf.name_scope(id):
            self.train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
            self.model = Network()

        self.global_brain = global_brain
        self._activate_weight()   # initialize weights for each workers.  

    def _activate_weight(self):
        x = np.zeros((1, NUM_STATES))
        self.model(x)

    def loss_func(self, s, a, r, p, v):
    
        # loss関数を定義します
        log_prob = tf.math.log(tf.math.reduce_sum( p*a, axis=1, keepdims=True) + 1e-10 )
        advantage = r - v
        loss_policy = - log_prob * tf.stop_gradient(advantage)    # stop_gradientでadvantageは定数として扱います
        loss_value = 0.5*tf.math.square(advantage)     # minimize value error
        entropy = 0.01*tf.math.reduce_sum( p*tf.math.log( p + 1e-10 ), axis=1, keepdims=True ) 
        return tf.math.reduce_mean(loss_policy + loss_value + entropy)

    def update_parameter(self):   # update gradients
    
        if len(self.train_queue[0]) < MIN_BATCH:    # データがたまっていない場合は更新しない
            return False

        s, a, r, s_, s_mask = self.train_queue  # train_queue is a list of list
        self.train_queue = [[], [], [], [], []]     # 使ったQueryは初期化する
        s = np.vstack(s)    # vstackはvertical-stackで縦方向に行列を連結、いまはただのベクトル転置操作
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)
        _, v = self.model.predict(s_)
        
        # N-1ステップあとまでの時間割引総報酬rに、Nから先に得られるであろう総報酬vに割引N乗したものを足します
        r = r + GAMMA_N*v*s_mask     # set v to 0 where s_ is terminal state

        # lossを取得
        with tf.GradientTape() as tape:
            p, v = self.model(s)
            loss = self.loss_func(s, a, r, p, v)

        # gradientの計算
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # global_brainの重みを更新
        self.global_brain.update_global_weight_params(gradients)

        return True

    def pull_global_parameter(self):    # global_brainの重みを引き出す (global→local)
        self.global_brain.pull_global_weight_params(self.model)

    def push_local_weight_params(self):     # global_brainに重みをコピーする（local→global)
        self.global_brain.push_local_weight_params(self.model)

    def train_push(self, s, a, r, s_):  # 学習用のデータを貯めておくキュー
        self.train_queue[0].append(s)
        self.train_queue[1].append(a)
        self.train_queue[2].append(r)

        if s_ is None:
            self.train_queue[3].append(NONE_STATE)
            self.train_queue[4].append(0.)
        else:
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)


class Agent:
    def __init__(self, id, global_brain, thread_type):
        self.brain = LocalBrain( id, global_brain )   # 行動を決定するための脳（ニューラルネットワーク）
        self.thread_type = thread_type
        self.memory = []        # s,a,r,s_の保存メモリ、used for n_step return
        self.Rewards = 0.             # 時間割引した、「いまからNステップ分あとまで」の総報酬R

        self.global_brain = global_brain
        
    def act(self, s):
    
        self.get_epsilon()  # Get epsilon

        if self.thread_type == 'learning':
            if random.random() < self.eps:
                action = random.randint(0, NUM_ACTIONS - 1)   # ランダムに行動
            else:
                s = np.array([s])
                #import pdb;  pdb.set_trace()  
                p, _ = self.brain.model.predict(s)

                # action = np.argmax(p)  # これだと確率最大の行動を、毎回選択
                action = np.random.choice(NUM_ACTIONS, p=p[0])
                # probability = p のこのコードだと、確率p[0]にしたがって、行動を選択
                # pにはいろいろな情報が入っていますが確率のベクトルは要素0番目

        else:
            s = np.array([s])
            p, _ = self.brain.model.predict(s)
            action = np.argmax(p)  # これだと確率最大の行動を、毎回選択
            # action = np.random.choice(NUM_ACTIONS, p=p[0])

        return action

    def advantage_push_local_brain(self, s, a, r, s_):   # advantageを考慮したs,a,r,s_をbrainに与える
        def get_sample(memory, n):  # advantageを考慮し、メモリからnステップ後の状態とnステップ後までのRを取得する関数
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]
            return s, a, self.Rewards, s_

        # one-hotコーディングにしたa_catsをつくり、、s,a_cats,r,s_を自分のメモリに追加
        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1
        self.memory.append((s, a_cats, r, s_))

        # 前ステップの「時間割引Nステップ分の総報酬R」を使用して、現ステップのRを計算
        self.Rewards = (self.Rewards + r * GAMMA_N) / GAMMA     # r0はあとで引き算している、この式はヤロミルさんのサイトを参照

        # advantageを考慮しながら、LocalBrainに経験を入力する
        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)
                self.Rewards = (self.Rewards - self.memory[0][2])/GAMMA
                self.memory.pop(0)

            self.Rewards = 0  # 次の試行に向けて0にしておく

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)
            self.Rewards = self.Rewards - self.memory[0][2]     # # r0を引き算
            self.memory.pop(0)

    def get_epsilon(self):  #  Make epsilon smaller for each trial.  
        if self.global_brain.frames >= EPS_STEPS or not self.thread_type == 'learning':   # ε-greedy法で行動を決定します 
            self.eps = EPS_END
        else:
            self.eps = EPS_START + self.global_brain.frames*(EPS_END - EPS_START)/EPS_STEPS  # linearly interpolate


class Environment:
    total_reward_vec = np.zeros(10)  # 総報酬を10試行分格納して、平均総報酬をもとめる
    count_trial_each_thread = 0     # count trial number for each workers.

    def __init__(self, id, thread_type, global_brain):
        self.id = id
        self.thread_type = thread_type   # learning or test
        self.env = gym.make(ENV)
        self.env._max_episode_steps = GOAL*2
        self.global_brain = global_brain
        
        self.agent = Agent(id, global_brain, thread_type)    # 環境内で行動するagentを生成

    def run(self):
        self.agent.brain.pull_global_parameter()    # global_brainの重みを自身ThreadのLocalBrainにコピー

        if (self.thread_type is 'test') and (self.count_trial_each_thread == 0):
            self.env.reset()
            #self.env = gym.wrappers.Monitor(self.env, directory="./{}/".format(start_datetime))
            #self.env = gnwrapper.Monitor(self.env, directory="./{}/".format(start_datetime))
                                            # force=True, video_callable=(lambda ep: ep % 1 == 0))    # 動画保存する場合

        # 全セッション内で共有
        s = self.env.reset()
        Rewards = 0
        step = 0
        while True:
            a = self.agent.act(s)   # 行動を決定
            s_, r, done, info = self.env.step(a)   # 行動を実施
            step += 1
            self.global_brain.frames += 1     # セッショントータルの行動回数をひとつ増やします

            r = 0
            if done:  # terminal state
                s_ = None
                if step < GOAL:
                    r = -1
                else:
                    r = 1

            # Advantageを考慮した報酬と経験を、localBrainにプッシュ
            self.agent.advantage_push_local_brain(s, a, r, s_)
            s = s_
            Rewards += r
            if done or (step % Tmax == 0):  # 終了時がTmaxごとに、parameterServerの重みを更新し、それをコピーする
                if not(self.global_brain.isLearned) and self.thread_type == 'learning':
                    self.agent.brain.update_parameter()   # Globalパラメータを更新
                    self.agent.brain.pull_global_parameter()    # global_brainの重みを自身ThreadのLocalBrainにコピー

            if done:
                self.total_reward_vec = np.hstack((self.total_reward_vec[1:], step))  # トータル報酬の古いのを破棄して最新10個を保持
                self.count_trial_each_thread += 1  # このスレッドの総試行回数を増やす
                break

        # 総試行数、スレッド名、今回の報酬を出力
        # print("スレッド："+self.id + "、試行数："+str(self.count_trial_each_thread) + "、今回のステップ:" + str(step)+"、平均ステップ："+str(self.total_reward_vec.mean()))
        print('Step: {} traial: {} local step: {} avg step: {} r'.format(self.id, self.count_trial_each_thread, step, self.total_reward_vec.mean()))

        if self.thread_type is 'test':
            return
          #return self.env.display()   # this is bugged.  
        # スレッドで平均報酬が一定を越えたら終了
        elif self.total_reward_vec.mean() >= GOAL:
            self.global_brain.isLearned = True
            time.sleep(3.0)     # この間に他のlearningスレッドが止まります
            self.agent.brain.push_local_weight_params()     # この成功したスレッドのパラメータをparameter-serverに渡します


class Worker:
    def __init__(self, id, thread_type, global_brain):
        self.environment = Environment(id, thread_type, global_brain)
        self.thread_type = thread_type

        self.global_brain = global_brain

    def run(self):
        if self.thread_type == 'learning':
            while True:
                if not self.global_brain.isLearned:
                    self.environment.run()
                else:
                    return True
        if self.thread_type == 'test':
          self.environment.run()


#------------------------------------------------------------------------------------------------------


threads = []
with tf.device("/cpu:0"):
    global_brain = GlobalBrain()    # Set up Global network.  
    for i in range(N_WORKERS):
        id = 'local_{}'.format(i)
        mode = "learning"    # "learning" or "test"
        #threads.append( Worker(thread_name=id, thread_type="learning", global_brain=global_brain) )
        threads.append( Worker( id, mode, global_brain) )
    # threads.append(Worker(thread_name="end_signal", thread_type="test", global_brain=global_brain))

# COORD = tf.train.Coordinator()  # TensorFlowでマルチスレッドにするための準備です
features = []
with PoolExecutor(max_workers=N_WORKERS) as executor:
    for worker in threads:
        job = lambda: worker.run()
        features.append(executor.submit(job))


# テストここから
W = Worker("test_thread", "test", global_brain)
W.run()