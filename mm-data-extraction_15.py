# -*- coding: utf-8 -*-
#=============================
# Created by Ken Nakatsukasa
# Apr. 28, 2021
#=============================

import os
import io
import sys
import datetime as dt
from dateutil.relativedelta import relativedelta
import json
import shutil
from distutils.dir_util import copy_tree
import sqlalchemy as sa
import psycopg2
import pandas as pd
import numpy as np
from isoweek import Week
import configparser
import decimal
from decimal import ROUND_DOWN
import time
import urllib.request
import ssl
import getopt
from envyaml import EnvYAML


class maimate_data_extraction:
    def __init__(self, options, args):
        try:
    
            self.time1 = time.time()
            self.dir_path = options[0][1]
            os.chdir(self.dir_path)
            self.curdir = os.getcwd()
            self.mkdir = options[1][1]
            self.cfg_file = options[2][1]
            self.cfg = EnvYAML(self.cfg_file)
            self.arg = int(args[0])
            #self.arg = self.cfg.get("settings.data_to_extract")
            
            # API endpoint
            self.APIENDPOINT = self.cfg.get("API_endpoint.url1")  
            
            # params for trade data
            #self.time2look_back = self.cfg.get("trade_info.days")
            self.num_agents = self.cfg.get("trade_info.num_agents")
            
            # postgresql
            self.db_connect_ai_engine = self.cfg.get("ai_db.read_replica")
            self.mysql_engine = sa.create_engine(self.db_connect_ai_engine)

            # tables
            self.agent_decisions_tbl = self.cfg.get("ai_db.tables.agent_decisions_tbl")  
            self.agent_evaluation_tbl = self.cfg.get("ai_db.tables.agent_evaluation_tbl")
            self.signal_stats_tbl = self.cfg.get("ai_db.tables.signal_stats_tbl")
            self.signal_stats_ranking_tbl = self.cfg.get("ai_db.tables.signal_stats_ranking_tbl")
            self.agent_ai_tbl = self.cfg.get("ai_db.tables.agent_ai_tbl")
            
            self.context = ssl.SSLContext(ssl.PROTOCOL_TLSv1)
            
            # mkdir
            self.strnow = str(dt.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0))
            self.yyyymm = self.strnow[0:7]
            self.yyyymmdd = self.strnow[0:10]
            self.time2look_back = self.strnow
            self.date = str(dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))[0:10]            
            if not os.path.exists(os.path.join(self.curdir, self.yyyymm)):
                os.makedirs(os.path.join(self.curdir, self.yyyymm))
            if not os.path.exists(os.path.join(self.mkdir, self.yyyymm)):
                os.makedirs(os.path.join(self.mkdir, self.yyyymm))
            self.strnow_next_month = str(dt.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) + relativedelta(months=1))
            self.yyyymm_next_month = self.strnow_next_month[0:7]
            self.yyyymmdd_next_month = self.strnow_next_month[0:10]
            
        except Exception as e:
            msg="=== Error at __init__(self):"
            raise Exception(msg,e)

    # arg = 1
    def get_decision(self):
        try:
        
            print("Getting agent_ai_decisions at ..............", dt.datetime.now().replace(microsecond=0) )
            str_query = "select * from " + self.agent_decisions_tbl + \
                        " where utc_datetime>='" + self.strnow + "'" + \
                        "   and utc_datetime<'"  + self.strnow_next_month + "'" + \
                        " order by agent_id, utc_datetime"
            pdf = pd.read_sql_query(str_query, self.mysql_engine)
            pdf.reset_index(drop=True, inplace=True)
            #pdf.to_csv(os.path.join(self.curdir, 'agent_ai_decisons.csv'))
            #pdf.to_csv(os.path.join(self.curdir, self.yyyymm, 'agent_ai_decisons.csv'))
            pdf.to_csv(os.path.join(self.mkdir, 'agent_ai_decisons.csv'))
            #pdf.to_csv(os.path.join(self.mkdir, self.yyyymm, 'agent_ai_decisons.csv'))
            
            '''
            str_query = "select * from " + self.agent_decisions_tbl + \
                        " where utc_datetime>='" + str(dt.datetime.now()-dt.timedelta(days=60)) + "'" + \
                        " order by agent_id, utc_datetime"
            pdf = pd.read_sql_query(str_query, mysql_engine)
            pdf.reset_index(drop=True, inplace=True)
            pdf.to_csv(os.path.join(self.curdir, 'agent_ai_decisons_long.csv'))
            '''
            
            print("Finished getting agent_ai_decisions at .....", dt.datetime.now().replace(microsecond=0)); print()
            
        except Exception as e:
            msg="=== Error at get_decision(self):"
            raise Exception(msg,e)
    
    # arg=2
    def get_evaluation(self):        
        try:
         
            print("Getting agent_ai_evaluation at .............", dt.datetime.now().replace(microsecond=0))
            str_query = "select * from " + self.agent_evaluation_tbl + \
                        " where signal_date>='" + self.strnow + "'" + \
                        "   and signal_date<'"  + self.strnow_next_month + "'" + \
                        " order by agent_id, signal_date"
            pdf = pd.read_sql_query(str_query, self.mysql_engine)
            pdf.reset_index(drop=True, inplace=True)
            #pdf.to_csv(os.path.join(self.curdir, 'agent_ai_evaluation.csv'))
            #pdf.to_csv(os.path.join(self.curdir, self.yyyymm, 'agent_ai_evaluation.csv'))
            pdf.to_csv(os.path.join(self.mkdir, 'agent_ai_evaluation.csv'))
            #pdf.to_csv(os.path.join(self.mkdir, self.yyyymm, 'agent_ai_evaluation.csv'))
            print("Finished getting agent_ai_evaluation at ....", dt.datetime.now().replace(microsecond=0)); print()
            
        except Exception as e:
            msg="=== Error at get_evaluation(self):"
            raise Exception(msg,e)
    
    # arg=3
    def get_stats(self):
        try:

            print("Getting signal_stats at ....................", dt.datetime.now().replace(microsecond=0) )
            str_query = "select * from " + self.signal_stats_tbl + \
                        " where base_date>='" + self.strnow + "'" + \
                        "   and base_date<'"  + self.strnow_next_month + "'" + \
                        " order by agent_id, base_date"
            self.data_stats = pd.read_sql_query(str_query, self.mysql_engine)
            self.data_stats.reset_index(drop=True, inplace=True)
            #self.data_stats.to_csv(os.path.join(self.curdir, 'signal_stats.csv'))
            #self.data_stats.to_csv(os.path.join(self.curdir, self.yyyymm, 'signal_stats.csv'))
            self.data_stats.to_csv(os.path.join(self.mkdir, 'signal_stats.csv'))
            #self.data_stats.to_csv(os.path.join(self.mkdir, self.yyyymm, 'signal_stats.csv'))
            print("Finished getting signal_stats at ...........", dt.datetime.now().replace(microsecond=0)); print()
            
        except Exception as e:
            msg="=== Error at get_stats(self):"
            raise Exception(msg,e)

    # arg=4
    def get_ranking(self):
        try:
        
            print("Getting signal_stats_ranking at ............", dt.datetime.now().replace(microsecond=0) )
            str_query = "select * from " + self.signal_stats_ranking_tbl + \
                        " where base_date>='" + self.strnow + "'" + \
                        "   and base_date<'"  + self.strnow_next_month + "'" + \
                        " order by agent_id, base_date"
            self.data_rank = pd.read_sql_query(str_query, self.mysql_engine)
            self.data_rank.reset_index(drop=True, inplace=True)
            #self.data_rank.to_csv(os.path.join(self.curdir, 'signal_stats_ranking.csv'))
            #self.data_rank.to_csv(os.path.join(self.curdir, self.yyyymm, 'signal_stats_ranking.csv'))
            self.data_rank.to_csv(os.path.join(self.mkdir, 'signal_stats_ranking.csv'))
            #self.data_rank.to_csv(os.path.join(self.mkdir, self.yyyymm, 'signal_stats_ranking.csv'))
            print("Finished getting signal_stats_ranking at ...", dt.datetime.now().replace(microsecond=0)); print()
                        
        except Exception as e:
            msg="=== Error at get_ranking(self):"
            raise Exception(msg,e)
            
    # arg=5
    def get_axon(self):
        try:
        
            # AXON data
            print("Getting user-agents at .....................", dt.datetime.now().replace(microsecond=0) )
            req = urllib.request.Request(url=self.APIENDPOINT, headers={'Content-Type':'application/json', 'User-Agent':'Mozilla/5.0'})
            f = urllib.request.urlopen(req, data={}, context=self.context)
            jsonurl = f.read()
            f.close
            #print(jsonurl)
            jsondata=json.loads(jsonurl)
            urllib.request.urlretrieve(jsondata["fileUrl"], os.path.join(self.dir_path, 'user-agents.json'))
            jdata = pd.read_json(r'./user-agents.json')
            jdata['creationTime'] = jdata['creationTime'].apply(lambda x: dt.datetime.utcfromtimestamp(x['epochSecond']).strftime('%Y-%m-%d %H:%M:%S'))
            jdata.to_csv(os.path.join(self.mkdir, 'user-agents.csv'), index=None)
            jdata_date = 'user-agents-' + self.date + '.csv'
            jdata.to_csv(os.path.join(self.mkdir, self.yyyymm, jdata_date), index=None)
            os.system('del user-agents.json') 
            print("Finished getting user-agents at ............", dt.datetime.now().replace(microsecond=0)); print()
            
            '''
            # url AXON data 
            urllib.request.urlretrieve('https://invast-mai-prod-mai-engine-user-agents-logs.s3.ap-northeast-1.amazonaws.com/json/user-agents.json', 'user-agents.json')
            jdata = pd.read_json(r'./user-agents.json')
            text = jdata['creationTime'].tolist()
            tlist = []
            for x in text:
                tlist.append( dt.datetime.utcfromtimestamp(x['epochSecond'] ).strftime('%Y-%m-%d %H:%M:%S')  )
            jdata['creationTime'] = tlist
            jdata.to_csv('./user-agents.csv')
            jdata.to_csv(os.path.join(os.getcwd(), 'user-agents.csv'))
            jdata.to_csv(os.path.join(os.getcwd(), self.yyyymm, 'user-agents.csv'))
            jdata.to_csv(os.path.join(mkdir, 'user-agents.csv'))
            jdata.to_csv(os.path.join(mkdir, self.yyyymm, 'user-agents.csv'))
            '''
            
        except Exception as e:
            msg="=== Error at get_axon(self):"
            raise Exception(msg,e)
      
    # arg=6
    def get_creation_time(self):
        try:
        
            # agent creation time
            print("Getting agent_ai at ........................", dt.datetime.now().replace(microsecond=0))
            str_query = "select agent_id, created_time_utc, instrument, technical_index, news_type, agent_appetite, latest_sim_type, latest_episode, latest_end_time_learning, latest_end_time_testing from " + self.agent_ai_tbl + " order by created_time_utc"
            pdf = pd.read_sql_query(str_query, self.mysql_engine)
            pdf.reset_index(drop=True, inplace=True)
                        
            # adding ranking, num_trades to agent_ai
            if len(self.data_rank)>0:
                base_date = self.data_rank.base_date.tail(1).values[0]
            else:
                base_date = self.strnow
            rank_data = self.data_rank[ self.data_rank['base_date']==base_date ]
            rank_data = rank_data[['agent_id', 'rank_number']]
            pdf = pd.merge( pdf, rank_data, on='agent_id', how='left' )
            stats_data = self.data_stats[ (self.data_stats['base_date']==base_date) & (self.data_stats['past_days']==365) ]
            stats_data = stats_data[['agent_id','total_signals']]
            pdf = pd.merge( pdf, stats_data, on='agent_id', how='left' ) 
            
            #pdf.to_csv(os.path.join(self.curdir, 'agent_ai.csv'))
            #pdf.to_csv(os.path.join(self.curdir, self.yyyymm, 'agent_ai.csv'))
            pdf.to_csv(os.path.join(self.mkdir, 'agent_ai.csv'))
            agent_ai_date = 'agent_ai-' + self.date + '.csv'
            pdf.to_csv(os.path.join(self.mkdir, self.yyyymm, agent_ai_date))
            
            print("Finished getting agent_ai at ...............", dt.datetime.now().replace(microsecond=0) ); print()
            
        except Exception as e:
            msg="=== Error at get_creation_time(self):"
            raise Exception(msg,e)
                        
    # arg=7
    def get_trade_data(self):
        try:
    
            # agent creation time
            print("Getting trade_data at ......................", dt.datetime.now().replace(microsecond=0))
            
            current_time = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            time2look_back = self.time2look_back   #current_time - dt.timedelta(days=self.time2look_back)

            str_query = "select agent_id, instrument, utc_datetime, realized_pnl, unrealized_pnl, bid_price_close, ask_price_close, ordered_price, current_position, trade_status from agent_ai_decisions " + \
                                           " where utc_datetime>= '" + str(time2look_back) + "' order by utc_datetime"
            df = pd.read_sql_query(str_query, self.mysql_engine)
            df.sort_values(['agent_id','utc_datetime'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            # total_trades
            data = pd.DataFrame( df['agent_id'].unique(), columns=['agent_id'] )
            data['past_days'] = time2look_back
            total_trades = df[(df['realized_pnl']!=0)].groupby(['agent_id'], as_index=False).size().to_frame('total_trades').reset_index()
            data = pd.merge( data, total_trades, on='agent_id', how='left')

            # total_pnl
            total_pnl = df[['agent_id','realized_pnl']].groupby( ['agent_id'], as_index=False).sum() 
            total_pnl.columns = ['agent_id', 'total_pnl']
            data = pd.merge( data, total_pnl, on='agent_id', how='left')

            '''
            # win_pct of profit
            nplus = df[(df['realized_pnl']>0)].groupby(['agent_id'], as_index=False).size().to_frame('nplus').reset_index()
            plusTotal = pd.merge( total_trades, nplus, on='agent_id', how='left')
            plusTotal.fillna(0, inplace=True)
            plusTotal['win_pct'] = plusTotal[['total_trades','nplus']].apply(lambda x: x[1]/x[0], axis=1 )
            plusTotal.drop(columns=['total_trades','nplus'], inplace=True)
            data = pd.merge( data, plusTotal, on='agent_id', how='left')

            # avg_profit 
            avg_pnl = df[(df['realized_pnl']>0)].groupby(['agent_id'], as_index=False).mean()
            avg_pnl = avg_pnl[['agent_id','realized_pnl']]
            avg_pnl.columns = ['agent_id','avg_profit']
            data = pd.merge( data, avg_pnl, on='agent_id', how='left')

            # avg_loss
            avg_loss = df[(df['realized_pnl']<0)].groupby(['agent_id'], as_index=False).mean()
            avg_loss = avg_loss[['agent_id','realized_pnl']]
            avg_loss.columns = ['agent_id','avg_loss']
            data = pd.merge( data, avg_loss, on='agent_id', how='left')

            # max_realized_pnl
            max_realized_pnl = df[['agent_id','realized_pnl']].groupby( ['agent_id'], as_index=False).max()
            max_realized_pnl.columns = ['agent_id','max_realized_pnl']
            data = pd.merge( data, max_realized_pnl, on='agent_id', how='left')
            '''
            
            # calculating drawdown 
            cum_pnl = df[['agent_id','realized_pnl']].copy()
            cum_pnl['csum_pnl'] = cum_pnl.groupby(['agent_id'])['realized_pnl'].cumsum()
            cum_pnl['cmax_pnl'] = cum_pnl.groupby(['agent_id'])['csum_pnl'].cummax()
            cum_pnl['dd'] = cum_pnl[['csum_pnl','cmax_pnl']].apply(lambda x: x[1]-x[0] , axis=1)
            cum_pnl_grouped = cum_pnl[['agent_id','dd']].groupby(['agent_id'], as_index=False).max()
            cum_pnl_grouped.columns = ['agent_id','max_dd']
            data = pd.merge( data, cum_pnl_grouped, on='agent_id', how='left')

            def fun(x):
                if x==0:  y='buy'
                elif x==1: y='nothing'
                else:  y='sell'
                return y

            # replace position with buy, sell string
            df['position'] = df[['current_position']].apply(lambda x: fun(x[0]), axis=1 )
            df.drop(columns=['current_position'], inplace=True)

            # get ranking based on number of trade & total_pnl
            data.fillna(0, inplace=True)
            data.sort_values(['total_trades','total_pnl'], ascending=[False,False], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data['ranking'] = data.index+1
            data = data.loc[ data['ranking']<=self.num_agents, :]

            trade_data = pd.merge( df, data, on='agent_id', how='inner')
            trade_data.sort_values(['ranking','utc_datetime'], inplace=True)

            #data.to_csv('./check.csv', index=False)
            #res.to_csv('./data.csv', index=False)

            data.to_csv(os.path.join(self.mkdir, 'trade_check.csv'), index=False)
            trade_data.to_csv(os.path.join(self.mkdir, 'trade_data.csv'), index=False)
            trade_data_date = 'trade_data-' + self.date + '.csv'
            trade_data.to_csv(os.path.join(self.mkdir, self.yyyymm, trade_data_date), index=False)
            
            print("Finished getting trade_data at .............", dt.datetime.now().replace(microsecond=0) ); print()
            
        except Exception as e:
            msg="=== Error at get_trade_data(self):"
            raise Exception(msg,e)

    def get_position(self):
        try:    
        
            print("Getting position_data at ...................", dt.datetime.now().replace(microsecond=0) )
            yesterday = str( (dt.datetime.now().replace(microsecond=0) - dt.timedelta(days=1)).date() )  
            str_query = "select agent_id, utc_datetime, current_position, realized_pnl, unrealized_pnl from " + self.agent_decisions_tbl + \
                        " where utc_datetime>='" + yesterday + "'" + \
                        " order by agent_id, utc_datetime"
            pdf = pd.read_sql_query(str_query, self.mysql_engine)
            pdf.reset_index(drop=True, inplace=True)
            pdf['num_users'] = np.nan
            pdf.to_csv(os.path.join(self.mkdir, 'position_data.csv'), index=False)
            pdf_date = 'position_data-' + self.date + '.csv'
            pdf.to_csv(os.path.join(self.mkdir, self.yyyymm, pdf_date), index=False)
            print("Finished getting posiiton_data at ..........", dt.datetime.now().replace(microsecond=0)); print()
                        
        except Exception as e:
            msg="=== Error at get_position(self):"
            raise Exception(msg,e)
            
def main():
    try:
        print("Process started at ", dt.datetime.now().replace(microsecond=0)); print()
        
        # "python main.py -d dir1 -m dir2 -c file x" or "python main.py --current_dir=dir1 --market_dir=dir2 --config_file=file x"
        # where x should be integer.  See below. 0 for all table extraction.
        # (note: you may want to use different name for args (args sometimes not working in python))
        options, args = getopt.getopt(sys.argv[1:], 'd:m:c:', ['current_dir=', 'market_dir=', 'config_file='] )

    except getopt.GetoptError as e:
        raise Exception(e)

    data_engine = maimate_data_extraction(options, args)
    if data_engine.arg == 0 or data_engine.arg == 1 :
        data_engine.get_decision()
    if data_engine.arg == 0 or data_engine.arg ==2 :
        data_engine.get_evaluation()
    if data_engine.arg == 0 or data_engine.arg ==3 :
        data_engine.get_stats()
    if data_engine.arg == 0 or data_engine.arg ==4 :
        data_engine.get_ranking()
    if data_engine.arg == 0 or data_engine.arg ==5 :
        data_engine.get_axon()
    if data_engine.arg == 0 or data_engine.arg ==6 :
        data_engine.get_creation_time()
    #if data_engine.arg == 0 or data_engine.arg == 7:
    #    data_engine.get_trade_data()
    if data_engine.arg == 0 or data_engine.arg == 8 :
        data_engine.get_position()
    if data_engine.arg not in [0,1,2,3,4,5,6,7,8]:
        print("Error in input format.  Exiting...")
        exit(1)
    
    print("Extraction ended. Execution time was %d seconds." %( (time.time()-data_engine.time1)  ) )

        
if __name__ == "__main__":
    main()

