# -*- coding: utf-8 -*-
import os
import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # DEBUG=>0, INFO=>1, WARNING=>2, ERROR=>3
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing
import scipy.signal
import datetime as dt
from pytz import timezone
import gc
import copy
import configparser

from ai_core.constants import EngineMode
from ai_core.modeling.data import Data_Agent

from ai_core.errors import maimateException
from eliot import log_call, preserve_context
import logging
class INV_AI():
    def __init__(self, engine_mode, settings, config, logger):
        self.engine_mode = engine_mode
        self.settings = settings
        self.config = config
        self.logger = logger
        
        # pandas
        pd.options.display.float_format = '{:.5f}'.format
        self._processed_data = pd.DataFrame([])
        self.output_agent_ai_decisions = pd.DataFrame([])

    def get_prepared_data_copy(self, start_hour_diff=0):
        start_hour_diff_column = "round_num_diff"
        columns_to_drop = [start_hour_diff_column]
        if self.engine_mode == EngineMode.CREATE or self.engine_mode == EngineMode.RETRAIN:
            pd_ttl_data = self._processed_data.loc[(self._processed_data[start_hour_diff_column]==start_hour_diff), :].copy()
            if "round_num" in pd_ttl_data.columns: columns_to_drop.append("round_num") #FIXME: Temporary fix to reuse old prepared data
            pd_ttl_data.drop(columns_to_drop, axis=1, inplace=True)
        elif self.engine_mode == EngineMode.DECISION:
            pd_ttl_data = self._processed_data.copy()
        
        #Check if data is not empty
        if pd_ttl_data.shape[0]==0:
            raise Exception("No data!! (engine_mode=%s, start_hour diff=%d)"%(self.engine_mode.name, start_hour_diff))
        return pd_ttl_data

    ############################
    # 関数:run
    @log_call(include_args=[],include_result=False)
    def run(self):
        try:
            num_of_round = 0
            np.random.seed(self.settings._random_seed)
            tf.random.set_seed(self.settings._random_seed)
            coordinator = tf.train.Coordinator()
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.settings.learning_rate,
                rho=self.settings.rmspropdecay,
                momentum=0.0,
                epsilon=1e-10,
                centered=False,
                name='RMSprop'
            )

            ################
            # Learning part
            if self.engine_mode == EngineMode.CREATE or self.engine_mode == EngineMode.RETRAIN:
                while (True):
                    num_of_round += 1
                    if self.settings.target_hour_change==True:
                        hour_diff =  self.settings._pattern_start_hour_diff[(num_of_round-1)]
                        start_time_learning_r=str(dt.datetime.strptime(self.settings.start_time_learning, '%Y-%m-%d %H:%M:%S') + dt.timedelta(hours=hour_diff))
                        end_time_learning_r=str(dt.datetime.strptime(self.settings.end_time_learning, '%Y-%m-%d %H:%M:%S') + dt.timedelta(hours=hour_diff))
                    else:
                        hour_diff = 0
                        start_time_learning_r=self.settings.start_time_learning
                        end_time_learning_r=self.settings.end_time_learning
                    
                    start_time = start_time_learning_r.replace('+00:00', '')
                    end_time = end_time_learning_r.replace('+00:00', '')
                    
                    # new kn edited    ,  loading data
                    #self._processed_data = pd.read_csv('C:/mnt_freq/ai_create_job/agents/data.csv', index_col=0)
                    #self._processed_data['utc_datetime'] = pd.to_datetime(self._processed_data['utc_datetime'])
                    prepared_data = self.get_prepared_data_copy(hour_diff)
                    

                    # JK commented out 2020/06/10
                    # tf.reset_default_graph()
                    
                    with tf.device("/cpu:0"):
                        np.random.seed(self.settings._random_seed)
                        # JK modified 2020/06/10
                        tf.random.set_seed(self.settings._random_seed)
                        # tf.set_random_seed(self.settings._random_seed)
                        
                        # JK modified 2020/06/10
                        # global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
                        # trainer = tf.train.RMSPropOptimizer(self.settings.learning_rate, self.settings.rmsprodecay)
                        if self.engine_mode == EngineMode.CREATE and num_of_round==1:
                            master_network = ac_network(
                                state_dim=self.settings.state_dim,
                                action_dim=self.settings.action_dim,
                                lstm_cell_units=self.settings.lstm_cell_units,
                                random_seed=self.settings._random_seed,
                                optimizer=optimizer,
                                engine_mode=self.engine_mode,
                                mode="new",
                                master_or_local="master",
                                model_dir=self.settings._model_dir,
                                old_model_dir=self.settings._old_model_dir                                
                            )
                        elif self.engine_mode == EngineMode.RETRAIN or self.engine_mode == EngineMode.DECISION:
                            master_network = ac_network(
                                state_dim=self.settings.state_dim,
                                action_dim=self.settings.action_dim,
                                lstm_cell_units=self.settings.lstm_cell_units,
                                random_seed=self.settings._random_seed,
                                optimizer=optimizer,
                                engine_mode=self.engine_mode,
                                mode="load",
                                master_or_local="master",
                                model_dir=self.settings._model_dir,
                                old_model_dir=self.settings._old_model_dir                                                                
                            )

                        # For Testing and visualization we only need one worker
                        if self.settings.number_of_workers==0:
                            self.num_workers = multiprocessing.cpu_count()
                        else:
                            self.num_workers = self.settings.number_of_workers

                        workers = []
                        # Create worker classes

                        for i in range(self.num_workers):
                            workers.append(worker(
                                            master_network=master_network,
                                            engine_mode=self.engine_mode,
                                            learning_or_execution="learning",
                                            settings=self.settings.copy(),
                                            config=self.config,
                                            i_name=i,
                                            optimizer=optimizer,
                                            num_of_round=num_of_round,
                                            random_act=True,
                                            start_time=start_time,
                                            end_time=end_time,
                                            preprocessed_data=prepared_data.copy()
                                        ))
                        # kn edited
                        # This is where the asynchronous magic happens.
                        # Start the "work" process for each worker in a separate thread.
                        worker_threads = []
                        for w in workers:
                            worker_work = lambda: w.work(coordinator)
                            t = threading.Thread(target=(worker_work))
                            t.start()
                            worker_threads.append(t)
                        coordinator.join(worker_threads)
                        tf.keras.backend.clear_session() 
                        
                    # 学習停止条件チェック 後で再検討
                    if (self.engine_mode == EngineMode.DECISION) or (num_of_round>=self.settings.max_number_of_round):
                        break
             
            ################
            # Inference part
            if self.engine_mode == EngineMode.CREATE or self.engine_mode == EngineMode.DECISION:
                start_time = self.settings.start_time_testing.replace('+00:00', '')
                end_time = self.settings.end_time_testing.replace('+00:00', '')
                prepared_data = self.get_prepared_data_copy(start_hour_diff=0)

                # kn edited, this is to change change trade frequency
                #ken_data1 = self.get_prepared_data_copy(start_hour_diff=0)
                #ken_data2 = self.get_prepared_data_copy(start_hour_diff=4)   
                #ken_data = pd.concat( [ ken_data1, ken_data2 ])
                #prepared_data = pd.dataframe()
                #prepared_data = ken_data.sort_values(by='utc_datetime')


                #with tf.device("/cpu:0"):  # kn edited 
                master_network = ac_network(
                    state_dim=self.settings.state_dim,
                    action_dim=self.settings.action_dim,
                    lstm_cell_units=self.settings.lstm_cell_units,
                    random_seed=self.settings._random_seed,
                    optimizer=optimizer,
                    engine_mode=self.engine_mode,
                    mode="load",
                    master_or_local="master",
                    model_dir=self.settings._model_dir,
                    old_model_dir=self.settings._old_model_dir
                )


                if False:   # kn edited   ,  use existing model.  In the input .py file, use new agent_id 
                    print('*********** use existing model *************') 
                    temp_dir = "C:/Users/ken_nakatsukasa/Desktop/BitBucket/agent_output/sample_agent/ai-164892b7c7b5-1568340815228"
                    if not os.path.isdir(temp_dir):
                        print('Directory does not exist!!!')
                        exit()
                    tmp_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.settings.learning_rate,rho=self.settings.rmspropdecay,
                        momentum=0.0,epsilon=1e-10,centered=False,name='RMSprop')
                    tmp_ckpt = tf.train.Checkpoint(optimizer=tmp_optimizer, model=master_network.model)
                    tmp_ckpoint_tf2 = tf.train.CheckpointManager(
                        checkpoint=tmp_ckpt,
                        directory=temp_dir,   
                        max_to_keep=1,
                        keep_checkpoint_every_n_hours=None,
                        checkpoint_name='model-tf2.1'  ) 
                    tmp_ckpt.restore(tmp_ckpoint_tf2.latest_checkpoint)
                    master_network.model = tmp_ckpt.model   #name master_network is defined just above


                self.num_workers = 1
                workers = []
                # Create worker classes
                self._processed_data.to_csv("_processed_data.csv")
                for i in range(self.num_workers):
                    workers.append(worker(
                                    master_network=master_network,
                                    engine_mode=self.engine_mode,
                                    learning_or_execution="execution",
                                    settings=self.settings,
                                    config=self.config,
                                    i_name=i,
                                    optimizer=optimizer,
                                    num_of_round=num_of_round,
                                    random_act=False,
                                    start_time=start_time,
                                    end_time=end_time,
                                    preprocessed_data=prepared_data
                                ))
                
                # This is where the asynchronous magic happens.
                # Start the "work" process for each worker in a separate thread.
                workers[0].work(coordinator)
                tf.keras.backend.clear_session() 
                
                # if num_of_round<self.settings.max_number_of_round: workers[0].csvout_paral.to_csv("worker0_trading_Completed.csv")
                # else: workers[0].csvout_paral.to_csv("worker0_trading_Incompleted.csv")

                # create data for agent_ai_decisons
                # JK revised 2020/07/03
                self.output_agent_ai_decisions = pd.DataFrame(workers[0].output_agent_ai_decisions.copy())
                # print("###### workers[0].output_agent_ai_decisions.copy()", workers[0].output_agent_ai_decisions.copy())
                # pd.DataFrame(workers[0].output_agent_ai_decisions.copy()).to_csv("csvout.csv")
                # pd.read_json(workers[0].output_agent_ai_decisions.copy()).to_csv("csvout.csv")
        
        except Exception as e:
            msg="=== Error @ run()"
            raise maimateException(msg,e)
        else:
            pass
        finally:
            pass


############################
# Class: ac_network
class ac_network():
    #@log_call(include_args=[],include_result=False)
    def __init__(
        self,
        state_dim,
        action_dim,
        lstm_cell_units,
        random_seed,
        optimizer,
        engine_mode,
        mode,
        master_or_local,
        model_dir,
        old_model_dir
    ):
        try:
            self.engine_mode = engine_mode
            self.optimizer = optimizer
            self.mode = mode
            self.model_dir_org = model_dir
            self.model_dir = model_dir
            self.old_model_dir = old_model_dir
            self.episode_count = 0
            self.lstm_cell_units = lstm_cell_units

            self.input_layer = tf.keras.Input(
                shape=(1,state_dim),
                batch_size=None,
                dtype=None,
                sparse=False,
                tensor=None,
                ragged=False,
                name="input_layer"
            )

            # LSTM layer to be defined, its shape needs to be (batch=1, timesteps=1, feature)
            self.lstm_layer = tf.keras.layers.LSTM(
                units=lstm_cell_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                kernel_initializer=tf.random_uniform_initializer(seed=random_seed),
                recurrent_initializer=None,
                bias_initializer='zeros',
                unit_forget_bias=True,
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.0,
                recurrent_dropout=0.0,
                implementation=2,
                return_sequences=False,
                return_state=True,
                go_backwards=False,
                stateful=False,
                time_major=False,
                unroll=False,
                trainable=True,
                name="lstm_layer"
            )(self.input_layer)

            # output layer to be defined
            self.policy_layer = tf.keras.layers.Dense(
                units=action_dim,
                activation='softmax',
                use_bias=None,
                kernel_initializer=tf.random_uniform_initializer(seed=random_seed),
                bias_initializer=None,
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name="policy_layer"
            )(self.lstm_layer[0])

            # output layer to be defined
            self.value_layer = tf.keras.layers.Dense(
                units=1,
                activation=None,
                use_bias=None,
                # kernel_initializer=tf.constant_initializer(value=1.0),
                kernel_initializer=tf.random_uniform_initializer(seed=random_seed),
                bias_initializer=None,
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name="value_layer"
            )(self.lstm_layer[0])

            # connect layers
            self.model = tf.keras.Model(
                inputs=[self.input_layer],
                outputs=[self.policy_layer, self.value_layer],
                name="maimate-model"
            )
            #print( self.model.summary()  )

            # ckpt creation
            if master_or_local=="master": self.model_ckpt_initialization()
            

        except Exception as e:
            msg= "=== Error @ ac_network.init()"
            raise maimateException(msg,e)
        else:
            pass
        finally:
            pass

    # function   : read_config_without_section
    def read_config_without_section(self, config_path):
        with open(config_path, 'r') as f:
            config_string = '[dummy_section]\n' + f.readline()
        config = configparser.ConfigParser()
        config.read_string(config_string)
        return {key: value.replace('\n', '') for key, value in config.items('dummy_section', raw=True)}

    # function   : model_ckpt_initialization
    # objective  : set paramters to master_network
    # parameters :
    #  - model   : "new" or "load"
    def model_ckpt_initialization(self):
        try:
            extracted_lstm_layer = self.model.get_layer(name="lstm_layer", index=None)
            extracted_policy_layer = self.model.get_layer(name="policy_layer", index=None)
            extracted_value_layer = self.model.get_layer(name="value_layer", index=None)
            if self.mode=="new":
                logging.info("---------------- checkpoint creation")
                self.ckpoint_tf2 = tf.train.CheckpointManager(
                    checkpoint=tf.train.Checkpoint(optimizer=self.optimizer, model=self.model),
                    directory=self.model_dir,
                    max_to_keep=1,
                    keep_checkpoint_every_n_hours=None,
                    checkpoint_name='model-tf2.1',
                )
                rt_save = self.ckpoint_tf2.save(checkpoint_number=0)
            elif self.mode=="load":
                # tf1.2 confirmation
                # _model_dir = 'G://004_Docker/Bitbucket-tf1.2/mai-engine-core/User_Agents/ai-0fae4ce9979f-1566039763742'
                #
                # tf2.1 confirmation
                # _model_dir = 'G://004_Docker/ai-engine-phase2-tensorflow2.0/ai-core/mai-mate2-ai-core/src/agents/d03'
                # ckpoint = tf.train.latest_checkpoint(_model_dir)
                # model_file_name = ckpoint.rsplit('/', 1)[1]
                # if model_file_name.find(os.sep)!=-1: model_file_name = model_file_name.rsplit(os.sep, 1)[1]
                # revised_model_path = os.path.join(_model_dir, model_file_name)
                # tf.train.list_variables(ckpt_dir_or_file=revised_model_path)
                # ckpoint = tf.train.latest_checkpoint(self.model_dir)

                # remove in future
                old_checkpoint_path = os.path.join(self.old_model_dir, 'checkpoint')
                checkpoint_path = os.path.join(self.model_dir, 'checkpoint')
                checkpoint_file = Path(checkpoint_path)
                model_dir = self.model_dir
                if not checkpoint_file.is_file():
                     checkpoint_path = old_checkpoint_path
                     model_dir = self.old_model_dir
                latest_ckpt = self.read_config_without_section(checkpoint_path)
                ckpoint = os.path.join(model_dir, latest_ckpt["model_checkpoint_path"].replace('"',''))
                logging.info("XXXXXXXXXXXXXXXXXXXXXXXXX ckpoint: %s"% ckpoint)
                try:
                    model_file_name = ckpoint.rsplit("/", 1)[1]
                except:
                    model_file_name = ckpoint.rsplit(os.sep, 1)[1]
                # print("XXXXXXXXXXXXXXXXXXXXXXXXX model_file_name(1):", model_file_name)
                if model_file_name.find(os.sep)!=-1: model_file_name = model_file_name.rsplit(os.sep, 1)[1]
                # print("XXXXXXXXXXXXXXXXXXXXXXXXX model_file_name(2):", model_file_name)
                revised_model_path = os.path.join(model_dir, model_file_name)
                # print("XXXXXXXXXXXXXXXXXXXXXXXXX revised_model_path:", revised_model_path)
                # tf2.1
                if model_file_name.startswith("model-tf2.1-"):
                    
                    logging.info("---------------- checkpoint loading tf2.1")
                    ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
                    logging.info("XXXXXXXXXXXXXXXXXXXXXXXXX ckpt: %s"%ckpt)
                    self.ckpoint_tf2 = tf.train.CheckpointManager(
                        checkpoint=ckpt,
                        directory=model_dir,
                        max_to_keep=1,
                        keep_checkpoint_every_n_hours=None,
                        checkpoint_name='model-tf2.1'
                    )
                    # print("XXXXXXXXXXXXXXXXXXXXXXXXX self.ckpoint_tf2:", self.ckpoint_tf2)
                    ckpt.restore(self.ckpoint_tf2.latest_checkpoint)
                    # print("XXXXXXXXXXXXXXXXXXXXXXXXX ckpt:", ckpt)
                    self.model = ckpt.model
                    self.episode_count = int(model_file_name.rsplit("-", 1)[1])
                    logging.info("XXXXXXXXXXXXXXXXXXXXXXXXX self.episode_count: %s"% self.episode_count)
                # tf.1.2
                elif model_file_name.startswith("model-"):
                    logging.info("---------------- checkpoint loading tf1.2")
                    # copy parameters in tf1.2 objects to tf2.1 objects
                    lstm_cell_kernel_row = extracted_lstm_layer.trainable_variables[0].shape[0]
                    #extracted_lstm_layer.set_weights(weights=[
                    #    tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/rnn/lstm_cell/kernel")[0:lstm_cell_kernel_row,:],
                    #    tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/rnn/lstm_cell/kernel")[lstm_cell_kernel_row:,:],
                    #    tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/rnn/lstm_cell/bias")
                    #])
                    # format conversion
                    # tensorflow1.2       : i, c, f, o
                    # tensorflow2.1(keras): i, f, c, o
                    lstm_kernel_convert_from_tf_to_keras = np.concatenate([
                        tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/rnn/lstm_cell/kernel")[:,0:self.lstm_cell_units],
                        tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/rnn/lstm_cell/kernel")[:,self.lstm_cell_units*2:self.lstm_cell_units*3],
                        tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/rnn/lstm_cell/kernel")[:,self.lstm_cell_units*1:self.lstm_cell_units*2],
                        tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/rnn/lstm_cell/kernel")[:,self.lstm_cell_units*3:]],
                        axis=1)
                    
                    lstm_bias_convert_from_tf_to_keras = np.concatenate([
                        tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/rnn/lstm_cell/bias")[0:self.lstm_cell_units],
                        tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/rnn/lstm_cell/bias")[self.lstm_cell_units*2:self.lstm_cell_units*3],
                        tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/rnn/lstm_cell/bias")[self.lstm_cell_units*1:self.lstm_cell_units*2],
                        tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/rnn/lstm_cell/bias")[self.lstm_cell_units*3:]],
                        axis=0)
                    extracted_lstm_layer.set_weights(weights=[
                        lstm_kernel_convert_from_tf_to_keras[0:lstm_cell_kernel_row,:],
                        lstm_kernel_convert_from_tf_to_keras[lstm_cell_kernel_row:,:],
                        lstm_bias_convert_from_tf_to_keras
                    ])

                    extracted_policy_layer.set_weights(weights=[
                        tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/fully_connected/weights")
                    ])
                    extracted_value_layer.set_weights(weights=[
                        tf.train.load_variable(ckpt_dir_or_file=revised_model_path, name="global/fully_connected_1/weights")
                    ])
                    # model saving in tf2.1 format
                    self.ckpoint_tf2 = tf.train.CheckpointManager(
                        checkpoint=tf.train.Checkpoint(optimizer=self.optimizer, model=self.model),
                        directory=self.model_dir_org,
                        max_to_keep=1,
                        keep_checkpoint_every_n_hours=None,
                        checkpoint_name='model-tf2.1'
                    )
                    rt_save = self.ckpoint_tf2.save(checkpoint_number=0)
                    self.model_dir = self.model_dir_org
            
            # parameter xh in LSTM should be zero in the case of model inference
            if (self.engine_mode == EngineMode.CREATE and self.mode=="load") or (self.engine_mode == EngineMode.DECISION and self.mode=="load"):
                extracted_lstm_layer.set_weights(weights=[
                    extracted_lstm_layer.trainable_variables[0].numpy(),
                    np.zeros(extracted_lstm_layer.trainable_variables[1].shape),
                    extracted_lstm_layer.trainable_variables[2].numpy()
                ])
            
            # print("##### extracted_lstm_layer.trainable_variables:", extracted_lstm_layer.trainable_variables)
        
        except Exception as e:
            msg="=== Error @ model_ckpt_initialization()"
            raise maimateException(msg,e)
        else:
            pass
        finally:
            pass

############################
# Class: worker
class worker():
    #@log_call(include_args=["num_of_round","i_name"],include_result=False)
    def __init__(self,
                master_network,
                engine_mode,
                learning_or_execution,
                settings,
                config,
                i_name,
                optimizer,
                num_of_round,
                random_act,
                start_time,
                end_time,
                preprocessed_data
    ):
        try:
            self.master_network = master_network
            self.engine_mode = engine_mode
            self.mode = self.master_network.mode
            self.learning_or_execution = learning_or_execution
            self.settings = settings
            self.config = config
            self.number = i_name
            self.name = "worker_" + str(i_name)
            self.state_dim = self.settings.state_dim
            self.action_dim = self.settings.action_dim
            self.optimizer = optimizer
            self.model_path = self.settings._model_dir
            self.old_model_path = self.settings._old_model_dir
            self.random_seed = self.settings._random_seed
            self.num_of_round = num_of_round
            self.random_act = random_act
            self.reward_factor = self.settings.reward_factor
            self.lstm_cell_units = self.settings.lstm_cell_units
            self.start_time = start_time
            self.end_time = end_time
            self.episode_rewards = []
            self.episode_states_realized = pd.DataFrame()
            self.episode_length = []
            self.episode_mean_values = []

            # Create the Lobal network to calculate gradient
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.settings.learning_rate,
                rho=self.settings.rmspropdecay,
                momentum=0.0,
                epsilon=1e-10,
                centered=False,
                name='RMSprop'
            )
            self.local_network = ac_network(
                                    state_dim=self.state_dim, 
                                    action_dim=self.action_dim, 
                                    lstm_cell_units=self.lstm_cell_units, 
                                    random_seed=self.random_seed,
                                    optimizer=self.optimizer,
                                    engine_mode=self.engine_mode,
                                    mode="new",
                                    master_or_local="local",
                                    model_dir=self.model_path,
                                    old_model_dir=self.old_model_path                                      
                                )
            self.copy_model_parameters(from_model=self.master_network, to_model=self.local_network)
            self.env = Data_Agent(
                    self.engine_mode, \
                    self.config, \
                    self.settings.copy(), \
                    self.name, \
                    self.num_of_round, \
                    start_time, \
                    end_time, \
                    self.settings.episode_minutes, \
                    self.settings.past_minute_to_see, \
                    self.settings.minutes_to_summarize, \
                    self.settings.number_of_rows_for_rolling, \
                    preprocessed_data)

        except Exception as e:
            msg="=== Error @ worker.init()"
            raise maimateException(msg,e)
        else:
            pass
        finally:
            pass
            
    ############################
    # train
    def train(self,
            gtape,
            episode_info_training, \
            episode_rewards_training, \
            episode_policy_training, \
            episode_values_training, \
            episode_actions_training):
        try:  
            episode_info_training_copy=copy.copy(episode_info_training)
            episode_rewards_training_copy=copy.copy(episode_rewards_training)
            episode_policy_training_copy=copy.copy(episode_policy_training)
            episode_actions_training_copy=copy.copy(episode_actions_training)
            del episode_info_training_copy[-1]
            del episode_rewards_training_copy[-1]
            del episode_policy_training_copy[-1]
            del episode_actions_training_copy[-1]

            # if self.name=="worker_0": pd.DataFrame(episode_info_training_copy).to_csv("episode_info_training_copy"+str(self.master_network.episode_count)+".csv")

            advantages = tf.math.add(episode_rewards_training_copy, tf.squeeze(tf.subtract(episode_values_training[1:], episode_values_training[:-1])))
            policy_responsible = tf.math.reduce_sum(tf.squeeze(episode_policy_training_copy)*episode_actions_training_copy, axis=1)
            entropy = -tf.reduce_sum(episode_policy_training_copy*tf.math.log(episode_policy_training_copy))
            policy_loss = tf.reduce_sum(tf.math.log(policy_responsible) * advantages)
            loss = tf.add(policy_loss, 0.01*entropy)
            #if self.name=='worker_0': print( loss.numpy()  )    # kn edited , 
            
            #'''
            # kn edited.  New loss function.  
            # Note that entropy and policy_responsible are using reduce_sum while others are reduce_mean
            # Note maimate 1.0 use discounting rewards 
            # if self.name=='worker_0':  import pdb; pdb.set_trace() 
            value_coef = 0.5
            gamma = 1.0   # Note discounted rewards are not used in maimate 1.5
            advantages = tf.math.add( episode_rewards_training_copy, tf.squeeze(tf.subtract(episode_values_training[1:], episode_values_training[:-1])))
            value_loss = tf.reduce_mean( 0.5*tf.math.square(advantages) )   # default=0.5
            policy_responsible = tf.math.reduce_sum( tf.squeeze(episode_policy_training_copy)*episode_actions_training_copy, axis=1, keepdims=False) 
            entropy = -tf.reduce_sum( episode_policy_training_copy*tf.math.log( tf.clip_by_value( episode_policy_training_copy, 1.0e-20, 1.0-1.0e-20) ))
            policy_loss = tf.reduce_mean( tf.math.log(policy_responsible + 1e-20) * tf.stop_gradient(advantages) )
            #policy_loss = tf.reduce_mean( tf.math.log(policy_responsible + 1e-20) * advantages )
            loss = value_coef*value_loss + policy_loss + 0.01*entropy
            #'''


            # calculate and apply gradients
            gradients = gtape.gradient(
                target=loss,
                sources=self.local_network.model.trainable_variables,
                output_gradients=None,
                unconnected_gradients=tf.UnconnectedGradients.NONE
            )
            gradients_clipped, global_norm = tf.clip_by_global_norm(t_list=gradients, clip_norm=5.0)
            gradients_clipped[0] = tf.where(tf.math.is_nan(gradients_clipped[0]), tf.zeros_like(gradients_clipped[0]), gradients_clipped[0])
            # if self.name=='worker_0': print("########## gradients_clipped:", gradients_clipped)
            
            if tf.reduce_sum(gradients_clipped[0]).numpy()!=0.0:
                rt = self.optimizer.apply_gradients(zip(gradients_clipped, self.master_network.model.trainable_variables))
                if self.name=='worker_0': logging.info("########## worker_0 trained")   # kn edited  
                
            self.master_network.episode_count += 1
            if self.master_network.episode_count >= 9999999: self.master_network.episode_count=0
            # if self.name=='worker_0': logging.info("########## worker_0 trained")   # kn edited  
        
        except Exception as e:
            msg="=== Error @ worker.train()"
            raise maimateException(msg,e)
        else:
            pass
        finally:
            pass
       
    ############################
    # work
    #@log_call(include_args=[],include_result=False)
    def work(self, coordinator):
        try:
            total_steps = 0
            #logging.info("Worker.work() number=%s round=%s thread=%s "%(self.number, self.num_of_round,threading.current_thread().name))

            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gtape:
                gtape.reset()
                if self.learning_or_execution=="learning":
                    gtape.watch(self.local_network.model.get_layer("lstm_layer").trainable_variables)
                    gtape.watch(self.local_network.model.get_layer("policy_layer").trainable_variables)
                    gtape.watch(self.local_network.model.get_layer("value_layer").trainable_variables)

                learning_terminal = False
                # print("##### state_info:", state_info)
                # print("##### type(state_info):", type(state_info))
                episode_info_training = []
                episode_info_training_out = []
                episode_rewards_training = []
                episode_values_training = []
                episode_policy_training = []
                episode_actions_training = []
                list_agent_ai_decisions = []

                kn_action=[1]  # kn edited  ,  policy analysis
                kn_price=[]
                #print( self.env.__dict__.keys() )  #display all existing variable in "Data_Agent" class in data.py  


                while (not coordinator.should_stop()) and (not learning_terminal):
                    exec_training = True
                    state_info, state_time, ep_len = self.env.get_state_info_for_model_with_current_position()
                                            
                    # kn edited  ,   plus 1 to ep_len for decision to avoid bug ,   len_ep
                    if self.engine_mode == EngineMode.DECISION:
                        ep_len += 1

                    # kn edited  , use ep_len condition for increasing episode duration ,   len_ep
                    while (len(episode_info_training)<self.settings.number_of_rows_for_model) and exec_training==True:
                    #while (len(episode_info_training)< ep_len-1 ) and exec_training==True:    

                        extracted_input_layer = self.local_network.model.get_layer(name="input_layer", index=None)
                        extracted_lstm_layer = self.local_network.model.get_layer(name="lstm_layer", index=None)
                        extracted_policy_layer = self.local_network.model.get_layer(name="policy_layer", index=None)
                        extracted_value_layer = self.local_network.model.get_layer(name="value_layer", index=None)

                        # state_info transformed into numpy.array as model input
                        model_input = np.expand_dims(np.expand_dims(np.array(list(state_info.values()), dtype=np.float32), axis=0), axis=0)
                        # if self.name=="worker_0": pd.DataFrame(list(state_info.values())).to_csv("state_info_"+str(state_time).replace(':','-')+"_"+str(self.master_network.episode_count)+".csv")

                        # model application
                        model_input_values = self.local_network.model.get_layer("input_layer", index=None).call(model_input)
                        # initial states should be zero in the case of first trade in an episode
                        if (len(episode_info_training)==0) or \
                            (self.engine_mode == EngineMode.CREATE and self.mode=="load") or \
                            (self.engine_mode == EngineMode.DECISION and self.mode=="load"):
                            model_hidden_h = tf.zeros([1,self.lstm_cell_units])
                            model_hidden_c = tf.zeros([1,self.lstm_cell_units])
                        # print("##### before model_hidden_h, model_hidden_c:", model_hidden_h.numpy()[0, 0:3], model_hidden_c.numpy()[0, 0:3])
                        model_lstm, model_hidden_h, model_hidden_c = self.local_network.model.get_layer("lstm_layer", index=None).call(model_input_values, initial_state=[model_hidden_h, model_hidden_c])
                        model_policy = self.local_network.model.get_layer("policy_layer", index=None)(model_lstm)
                        model_value = self.local_network.model.get_layer("value_layer", index=None)(model_lstm)
                        # print("##### after model_hidden_h, model_hidden_c:", model_hidden_h.numpy()[0, 0:3], model_hidden_c.numpy()[0, 0:3])
                        # print("++++++++++++++++++++++++++++ self.random_act:", self.random_act)
                        if self.random_act==True:
                            action_to_take = np.searchsorted(
                                np.cumsum([1.0/3.0, 1.0/3.0, 1.0/3.0]),
                                np.random.rand(1)*1.0)
                        else:
                            #action_to_take = tf.math.argmax(input=model_policy, axis=1).numpy()[0]

                            # kn edited  ,  policy analysis
                            if self.env.preprocessed_data.columns[6] == 'PRICE_RSI1440_mean':
                                pass
                            elif self.env.preprocessed_data.columns[6] == 'PRICE_MA1440_mean':
                                pass
                            elif self.env.preprocessed_data.columns[6] == 'PRICE_BBANDS_UPPER0_mean':
                                pass
                            else:
                                pass


                            # kn edited  ,  policy analysis 
                            #kn_mean = np.mean( [abs(state_info['PRICE_RSI1440_mean']),  abs(state_info['PRICE_RSI4320_mean']),  abs(state_info['PRICE_RSI7200_mean']) ]  )
                            kn_mean = 0
                            '''
                            if len(kn_action)>0:
                                kn_diff =  np.mean( [abs(state_info['PRICE_RSI1440_mean']-kn_price[-1][0]),  abs(state_info['PRICE_RSI4320_mean']-kn_price[-1][1]),  abs(state_info['PRICE_RSI7200_mean']-kn_price[-1][2]) ] ) 
                                if kn_diff > 0.8:
                                    action_to_take = tf.math.argmax(input=model_policy, axis=1).numpy()[0]
                                else:
                                    action_to_take = kn_action[-1]
                            else:
                                action_to_take = tf.math.argmax(input=model_policy, axis=1).numpy()[0]
                            '''
                            #'''
                            if kn_mean>=0.0:
                                action_to_take = tf.math.argmax(input=model_policy, axis=1).numpy()[0]
                            else:
                                action_to_take = kn_action[-1]
                            #'''

                            # kn edited  ,  action=1 when unrealized_pnl greater than 500 during testing phase (in creation) , this is for inference part
                            #if  "json_agent_ai_decisions" in locals() and json_agent_ai_decisions['unrealized_pnl']*100>250 : 
                                #action_to_take = 1
                                #print("force to trade ******", json_agent_ai_decisions['unrealized_pnl']*100 )
                            #print(self.config.unrealized_pnl_dict, self.config.pips_multiplier_dict,  self.settings.cpair  )
                        
                        # kn edited  ,  polycy analysis
                        kn_action.append(action_to_take)  
                        #kn_price.append(  (state_info['PRICE_RSI1440_mean'],  state_info['PRICE_RSI4320_mean'],  state_info['PRICE_RSI7200_mean'])  )


                        actions = np.zeros(self.settings.action_dim)
                        actions[action_to_take] = 1.0

                        # print("++++++++++++++++++++++++++++ self.env.pd_ttl_data.shape[0]:", self.env.pd_ttl_data.shape[0])
                        # print("++++++++++++++++++++++++++++ self.env.simulation_index+1:", self.env.simulation_index+1)
                        if self.env.pd_ttl_data.shape[0]>self.env.simulation_index+1 or self.engine_mode == EngineMode.DECISION:
                            # if self.name == 'worker_0': print("!!!!!!!!!!!!!!!!!!!! self.env.pd_ttl_data.shape[0]:", self.env.pd_ttl_data.shape[0]) 
                            # if self.name == 'worker_0': print("!!!!!!!!!!!!!!!!!!!! self.env.simulation_index+1:", self.env.simulation_index+1) 
                            # if self.name == 'worker_0': print("!!!!!!!!!!!!!!!!!!!! self.env.pd_ttl_data.loc[self.env.simulation_index, utc_datetime]:", self.env.pd_ttl_data.loc[self.env.simulation_index, 'utc_datetime']) 
                            # if self.name == 'worker_0': print("!!!!!!!!!!!!!!!!!!!! self.env.pd_ttl_data.loc[self.env.simulation_index+1, utc_datetime]:", self.env.pd_ttl_data.loc[self.env.simulation_index+1, 'utc_datetime']) 
                            state_info_next, \
                            state_time_next, \
                            reward, \
                            realized_pnl, \
                            executed_action, \
                            terminal, \
                            learning_terminal, \
                            model_input_info, \
                            latest_candle_stick_prices, \
                            json_agent_ai_decisions = self.env.exec_trading_simulation_to_learn(action=action_to_take)

                            episode_info_training.append([state_info, state_time, action_to_take, reward, state_info_next, terminal, realized_pnl])
                            episode_info_training_out.append([state_info, state_time, action_to_take, reward, state_info_next, terminal, realized_pnl])
                            episode_rewards_training.append(reward)
                            episode_policy_training.append(model_policy)
                            episode_values_training.append(model_value)
                            episode_actions_training.append(actions.tolist())

                            # JK added 2020/07/03
                            if self.name == 'worker_0':
                                json_agent_ai_decisions['agent_decision_id'] = "to_be_updated"
                                json_agent_ai_decisions['agent_id'] = "to_be_updated"
                                json_agent_ai_decisions['instrument'] = "to_be_updated"
                                json_agent_ai_decisions['news_type'] = "to_be_updated"
                                json_agent_ai_decisions['agent_appetite'] = "to_be_updated"
                                json_agent_ai_decisions['agent_version'] = self.config.agent_version_in_ini
                                if self.engine_mode==EngineMode.CREATE:
                                    json_agent_ai_decisions['sim_type'] = "learning"
                                else:
                                    json_agent_ai_decisions['sim_type'] = self.learning_or_execution
                                json_agent_ai_decisions['model_dir'] = os.getcwd()
                                json_agent_ai_decisions['num_of_workers'] = self.settings.number_of_workers
                                json_agent_ai_decisions['num_of_round'] = self.num_of_round
                                json_agent_ai_decisions['episode'] = self.master_network.episode_count
                                json_agent_ai_decisions['input_info'] = [state_info, state_time, action_to_take, reward, state_info_next, terminal, realized_pnl, model_policy]
                                json_agent_ai_decisions['start_time'] = self.start_time
                                json_agent_ai_decisions['end_time'] = self.end_time
                                if self.learning_or_execution=="learning" or self.engine_mode==EngineMode.CREATE:
                                    json_agent_ai_decisions['start_time_learning'] = self.start_time
                                    json_agent_ai_decisions['end_time_learning'] = self.end_time
                                    json_agent_ai_decisions['start_time_testing'] = dt.datetime(1900,1,1,0,0,0)
                                    json_agent_ai_decisions['end_time_testing'] = dt.datetime(1900,1,1,0,0,0)
                                else:
                                    json_agent_ai_decisions['start_time_learning'] = dt.datetime(1900,1,1,0,0,0)
                                    json_agent_ai_decisions['end_time_learning'] = dt.datetime(1900,1,1,0,0,0)
                                    json_agent_ai_decisions['start_time_testing'] = self.start_time
                                    json_agent_ai_decisions['end_time_testing'] = self.end_time
                                list_agent_ai_decisions.append(json_agent_ai_decisions)

                        if self.name == 'worker_0':
                            logging.info("########## len(episode): %s    state_time: %s    state_time_next: %s"%(len(episode_info_training), state_time, state_time_next))
                        
                        if (state_time==state_time_next):
                            exec_training = False

                        state_info = state_info_next
                        state_time = state_time_next

                    # when episode is ready, do training
                    if self.learning_or_execution=="learning" and exec_training==True:
                        self.train(
                            gtape,
                            episode_info_training,
                            episode_rewards_training,
                            episode_policy_training,
                            episode_values_training,
                            episode_actions_training)

                    # gradient recording
                    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXX-1 self.learning_or_execution:", self.learning_or_execution)
                    gtape.reset()
                    self.copy_model_parameters(from_model=self.master_network, to_model=self.local_network)
                    if self.learning_or_execution=="learning":
                        gtape.watch(self.local_network.model.get_layer("lstm_layer").trainable_variables)
                        gtape.watch(self.local_network.model.get_layer("policy_layer").trainable_variables)
                        gtape.watch(self.local_network.model.get_layer("value_layer").trainable_variables)

                    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXX-2")
                    # training episode hand over
                    # episode_info_training_out.pop(-1)
                    episode_info_training = []
                    episode_rewards_training = []
                    episode_values_training = []
                    episode_policy_training = []
                    episode_actions_training = []

                    if self.name == 'worker_0' and self.learning_or_execution=="learning":
                        # print("XXXXXXXXXXXXXXXXXXXXXXXXXXX-3 self.master_network.episode_count:", self.master_network.episode_count)
                        # print("XXXXXXXXXXXXXXXXXXXXXXXXXXX-3 self.master_network.ckpoint_tf2:", self.master_network.ckpoint_tf2)
                        rt_save = self.master_network.ckpoint_tf2.save(checkpoint_number=self.master_network.episode_count)
                    
                    if self.name == 'worker_0' and self.learning_or_execution=="execution":
                        # print("XXXXXXXXXXXXXXXXXXXXXXXXXXX-4")
                        # JK revised 2020/07/03
                        self.output_agent_ai_decisions = list_agent_ai_decisions.copy()
                    
                    if self.name == 'worker_0':
                        # print("XXXXXXXXXXXXXXXXXXXXXXXXXXX-5")
                        # print("| Round:" + str(self.num_of_round) + " | " + self.name + " | utc_datetime: " + str(min(self.env.pd_ttl_data['utc_datetime'])) + "  ~  " + str(max(self.env.pd_ttl_data['utc_datetime'])) + " | Episode", self.master_network.episode_count)
                        logging.info("| Round: %s | %s | utc_datetime: %s  ~  %s | Episode %s"%(str(self.num_of_round), self.name, str(min(self.env.pd_ttl_data['utc_datetime'])) , str(max(self.env.pd_ttl_data['utc_datetime'])), self.master_network.episode_count))
        
        except Exception as e:
            msg="=== Error @ worker.work()"
            raise maimateException(msg,e)
        else:
            pass
        finally:
            pass

    ############################
    # copy_model_parameters
    # Copies one set of variables to another.
    # Used to set worker network parameters to those of global network
    @staticmethod
    def copy_model_parameters(from_model, to_model):
        try:
            to_model.model.set_weights(weights=from_model.model.get_weights())
    
        except Exception as e:
            msg= "=== Error @ copy_model_parameters()"
            raise maimateException(msg,e)
        else:
            pass
        finally:
            pass

    ############################
    # random_pick
    # @staticmethod
    # def random_pick(weights, n_picks):
    #     try:
    #         t = np.cumsum([1.0/3.0, 1.0/3.0, 1.0/3.0])
    #         s = 1.0
    #         return np.searchsorted(t, np.random.rand(n_picks)*s)
    #  
    #     except Exception as e:
    #         msg="=== Error @ random_pick()"
    #         raise maimateException(msg,e)
    #     else:
    #         pass
    #     finally:
    #         pass

    ############################
    # 関数:discounting
    # Discounting function used to calculate discounted returns
    # @staticmethod
    # def discounting(x, gamma):
    #     try:
    #         return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    # 
    #     except Exception as e:
    #         msg="=== Error @ discounting()"
    #         raise maimateException(msg,e)
    #     else:
    #         pass
    #     finally:
    #         pass
