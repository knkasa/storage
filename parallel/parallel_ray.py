import csv
import sys
sys.path.append('C:/Users/ken_nakatsukasa/Desktop/BitBucket1968_test14/mai-mate2-ai-ranking-job/src')
sys.path.append('C:/Users/ken_nakatsukasa/Desktop/BitBucket1968_test14/mai-mate2-ai-data-job/src')

from ai_core.job import Job
from ai_core.constants import EngineMode, ExecEnv

import json, sys
from eliot import log_call, to_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate
import gc
import datetime as dt
import time

import ray


#------------ edit here -------------------------------------------------------------

#**** 新しくテクニカルをテストする場合は、output_agent_ai_decisions.csvなどをフォルダから移動しておくこと。*****
core_dir = ('C:/Users/ken_nakatsukasa/Desktop/BitBucket1968_test14/mai-mate2-ai-core/src')

# directory of "ai_info.csv"
df = pd.read_csv('C:/Users/ken_nakatsukasa/Desktop/GBPUSD_RSI_sentiment/ai_info.csv')
df = df[(df['symbol']=='GBP/USD') & (df['technical_index']=='HIBRID_BBAND')]   #.head(5)

# directory of output files (see env.yml)
output_dir = 'C:/Users/ken_nakatsukasa/Desktop/BitBucket1968_test14/output/'  

# https://mrjbq7.github.io/ta-lib/funcs.html
# MA, KAMA, BAND, PRICE, RSI, MACD, ROC, MOM, ENV
technical = 'RSI'

one_year = False   # True if you only want result for 1 year

interp_kind = 'linear'  # choose "linear" or "cubic"    

first_run = True   
third_run = True
  
test_model = False 
test_upper = 0.6  ;   test_lower = -1.3 ;

errlog_dir = 'C:\logs'   # should be same as the one in env.yml

#------------------------------------------------------------------------------------


# parameter sets
upper = np.linspace( 2, -0.0, 5 )     #  np.linspace( 2, -0.0, 5 )   
lower = np.linspace( 0.0, -2, 5 )     #  np.linspace( 0.0, -2, 5 )    

num_process = 5   # numper of process (do not change)

time2sleep = 5  # time to wait before running next job

error_dir = errlog_dir + '\maimate*'

cur = df.loc[df.index.values[0],'symbol']
agent_tech = df.loc[df.index.values[0],'technical_index']

# checking output file
output_check = os.listdir(output_dir) 
if len(output_check)>0:
    output_name = output_check[0]
    output_list = output_check[0].split("-")
    cur_name = cur[0:3] + cur[4:7]
    if (cur_name != output_list[2]) or (agent_tech != output_list[3]) or (technical != output_list[1]) :
        print("wrong agent technical or currency.  Exiting...")
        exit()

# checking config for third only run
if first_run==False and third_run==True:
    os.chdir(output_dir)
    file_check = open("../result2.txt", "r")
    list_check = file_check.read().split(" ")
    tech_check = list_check[0]
    cur_check = list_check[7]
    data_check = list_check[9]
    if tech_check != technical or cur_check != cur or data_check != agent_tech:
        print("Initial config not matching!!  Exiting...")
        exit()

num_content = round(len(df)/num_process)
dic = {}
for n in range(num_process):
    dic['id'+str(n)] = df.head(num_content)
    df.drop( df.head(num_content).index, axis=0, inplace=True )

    
@ray.remote
def fun0(new_model):
    #os.system('del ' + error_dir)
    #time.sleep(time2sleep)
    df = dic['id0']
    ilist = df.index.tolist()
    for n in ilist:
        gc.collect()
        xid = 'test' + str(n) + '-' + new_model['tech'] + '-' + df.loc[n,'symbol'][0:3]+df.loc[n,'symbol'][4:7] + '-' + df.loc[n,'technical_index'] + '-' + df.loc[n,'agent_id'][0:8]    
        content = { "agent_id": xid,
                     "nickname": "test_ken",
                     "instrument": df.loc[n,'symbol'][0:3]+df.loc[n,'symbol'][4:7],
                     "agent_appetite": "RISK_HATER",
                     "technical_index": df.loc[n,'technical_index'],
                     "news_type": df.loc[n,'news_type'] }
        json_content = json.dumps(content)
        job = Job(env, engine_mode, json_content, df.loc[n,'agent_id'], new_model)
        job.run()
        #time.sleep(time2sleep)
        #os.system('del ' + error_dir)
    return 0
        
@ray.remote
def fun1(new_model):
    #os.system('del ' + error_dir)
    #time.sleep(time2sleep)
    df = dic['id1']
    ilist = df.index.tolist()
    for n in ilist:
        gc.collect()
        xid = 'test' + str(n) + '-' + new_model['tech'] + '-' + df.loc[n,'symbol'][0:3]+df.loc[n,'symbol'][4:7] + '-' + df.loc[n,'technical_index'] + '-' + df.loc[n,'agent_id'][0:8]    
        content = { "agent_id": xid,
                     "nickname": "test_ken",
                     "instrument": df.loc[n,'symbol'][0:3]+df.loc[n,'symbol'][4:7],
                     "agent_appetite": "RISK_HATER",
                     "technical_index": df.loc[n,'technical_index'],
                     "news_type": df.loc[n,'news_type'] }
        json_content = json.dumps(content)
        job = Job(env, engine_mode, json_content, df.loc[n,'agent_id'], new_model)
        job.run()
        time.sleep(time2sleep)
        #os.system('del ' + error_dir)
    return 1

@ray.remote
def fun2(new_model):
    #os.system('del ' + error_dir)
    #time.sleep(time2sleep)
    df = dic['id2']
    ilist = df.index.tolist()
    for n in ilist:
        gc.collect()
        xid = 'test' + str(n) + '-' + new_model['tech'] + '-' + df.loc[n,'symbol'][0:3]+df.loc[n,'symbol'][4:7] + '-' + df.loc[n,'technical_index'] + '-' + df.loc[n,'agent_id'][0:8]    
        content = { "agent_id": xid,
                     "nickname": "test_ken",
                     "instrument": df.loc[n,'symbol'][0:3]+df.loc[n,'symbol'][4:7],
                     "agent_appetite": "RISK_HATER",
                     "technical_index": df.loc[n,'technical_index'],
                     "news_type": df.loc[n,'news_type'] }
        json_content = json.dumps(content)
        job = Job(env, engine_mode, json_content, df.loc[n,'agent_id'], new_model)
        job.run()
        time.sleep(time2sleep)
        #os.system('del ' + error_dir)
    return 2

@ray.remote
def fun3(new_model):
    #os.system('del ' + error_dir)
    #time.sleep(time2sleep)
    df = dic['id3']
    ilist = df.index.tolist()
    for n in ilist:
        gc.collect()
        xid = 'test' + str(n) + '-' + new_model['tech'] + '-' + df.loc[n,'symbol'][0:3]+df.loc[n,'symbol'][4:7] + '-' + df.loc[n,'technical_index'] + '-' + df.loc[n,'agent_id'][0:8]    
        content = { "agent_id": xid,
                     "nickname": "test_ken",
                     "instrument": df.loc[n,'symbol'][0:3]+df.loc[n,'symbol'][4:7],
                     "agent_appetite": "RISK_HATER",
                     "technical_index": df.loc[n,'technical_index'],
                     "news_type": df.loc[n,'news_type'] }
        json_content = json.dumps(content)
        job = Job(env, engine_mode, json_content, df.loc[n,'agent_id'], new_model)
        job.run()
        time.sleep(time2sleep)
        #os.system('del ' + error_dir)
    return 3

@ray.remote
def fun4(new_model):
    #os.system('del ' + error_dir)
    #time.sleep(time2sleep)
    df = dic['id4']
    ilist = df.index.tolist()
    for n in ilist:
        gc.collect()
        xid = 'test' + str(n) + '-' + new_model['tech'] + '-' + df.loc[n,'symbol'][0:3]+df.loc[n,'symbol'][4:7] + '-' + df.loc[n,'technical_index'] + '-' + df.loc[n,'agent_id'][0:8]    
        content = { "agent_id": xid,
                     "nickname": "test_ken",
                     "instrument": df.loc[n,'symbol'][0:3]+df.loc[n,'symbol'][4:7],
                     "agent_appetite": "RISK_HATER",
                     "technical_index": df.loc[n,'technical_index'],
                     "news_type": df.loc[n,'news_type'] }
        json_content = json.dumps(content)
        job = Job(env, engine_mode, json_content, df.loc[n,'agent_id'], new_model)
        job.run()
        time.sleep(time2sleep)
        #os.system('del ' + error_dir)
    return 4


def get_pnls():
    os.chdir(output_dir)
    files = os.listdir()  
    pnl_list = []
    dd_list = []
    num_trade_list = []
    csvdata = '/output_agent_ai_decisions.csv'
    for m, x in enumerate(files):   

        m += 1
        dir = output_dir + x + csvdata
        df = pd.read_csv( dir, delimiter=',' )
        
        if one_year:
            endtime_learn = df['end_time'].tolist()[-1]
            endtime_learn = dt.datetime.strptime(endtime_learn, '%Y-%m-%d %H:%M:%S')
            last_1year = endtime_learn - dt.timedelta(days=365) 
            df = df[ df['utc_datetime'] > str(last_1year) ].copy()  #only need 1yr of data for cum_pnl
        
        num_row = df.shape[0]
        xpnl = df['realized_pnl'].to_numpy()
        xpnl = np.cumsum(xpnl)
        xpnl = xpnl[0:num_row]
                        
        # max dd
        xdd = np.maximum.accumulate(xpnl)
        dd = np.max(xdd-xpnl)
        dd_list.append(dd)

        # sum pnl
        pnl = np.sum(df['realized_pnl'].to_numpy())
        pnl_list.append(pnl)
        
        num_trade = len( df[ df['realized_pnl']!=0.0 ] )
        num_trade_list.append(num_trade)
                
    return pnl_list, dd_list, num_trade_list
        
def del_logs():
    log_files = os.listdir(errlog_dir)  
    count = 1
    while len(log_files)!=0 and count<=20:
        os.system('del ' + error_dir)
        time.sleep(1)
        log_files = os.listdir(errlog_dir)
        count += 1

        
#-------------------------------------------------------------------------------------------------
    
env = ExecEnv.LOCAL
engine_mode = EngineMode.CREATE

pnl_mat = np.zeros([len(upper),len(lower)])
dd_mat = np.zeros([len(upper),len(lower)])
trade_mat = np.zeros([len(upper),len(lower)])

total_run=0
for i, x1 in enumerate(upper):
    for j, x2 in enumerate(lower):
        if x1>x2 and (x1-x2)>=1.0 and (x1-x2)<=2.4:
            total_run+=1
            print( x1, x2 )


#---------- getting first pnl matrix --------------------------------------------------------------

if first_run:
    l = 0
    for i, x1 in enumerate(upper):
        for j, x2 in enumerate(lower):
            
            if x1>x2 and (x1-x2)>=1.0 and (x1-x2)<=2.4:
            
                new_model = { "tech":technical, "upper":x1, "lower":x2 }
                l += 1
                print(); print(" *************** param = " + str(l) + " of " + str(total_run) ); print()
                
                #del_logs()
                #time.sleep(l)
                os.chdir(core_dir)
                ray.init( num_cpus=4 )
                time.sleep(5)
                print(" check1" )
                ray0 = fun0.remote(new_model)
                time.sleep(5)
                print(" check2" )
                ray1 = fun1.remote(new_model)
                time.sleep(5)
                print(" check3" )
                ray2 = fun2.remote(new_model)
                time.sleep(5)
                print(" check4" )
                ray3 = fun3.remote(new_model)
                time.sleep(5)
                print(" check5" )
                ray4 = fun4.remote(new_model)
                print(" chech x " )
                time.sleep(5)
                ray_res = ray.get([ray0,ray1,ray2,ray3,ray4])
                ray.shutdown()
                del_logs()
                
                gc.collect()
                print()
                
                pnl_list, dd_list, num_trade_list = get_pnls()
                                        
                pnl_mat[i,j] = np.mean(pnl_list)
                dd_mat[i,j] = np.mean(dd_list)
                trade_mat[i,j] = np.mean(num_trade_list)
                print(); print( pnl_mat ); print()
                
                # create log
                final = " param = " + str(l) + " of " + str(total_run)
                if not os.path.exists("../log"):
                    os.makedirs("../log")
                log_dir = "../log/param" + str(l) + "of" + str(total_run) + ".txt"
                f = open(log_dir, 'w')
                f.write( final )
                f.close()

                
    #-------- Getting normal result for comparison -----------------------------------
    new_model = { "tech":technical, "upper":0., "lower":-0. }

    os.chdir(core_dir)
    ray.init()
    time.sleep(2)
    print(" check1" )
    ray0 = fun0.remote(new_model)
    time.sleep(2)
    print(" check2" )
    ray1 = fun1.remote(new_model)
    time.sleep(2)
    print(" check3" )
    ray2 = fun2.remote(new_model)
    time.sleep(2)
    print(" check4" )
    ray3 = fun3.remote(new_model)
    time.sleep(2)
    print(" check5" )
    ray4 = fun4.remote(new_model)
    print(" chech x " )
    time.sleep(2)
    ray_res = ray.get([ray0,ray1,ray2,ray3,ray4])
    ray.shutdown()
    del_logs()
    gc.collect()
    print()

    pnl_list, dd_list, num_trade_list = get_pnls()

    pnl0 = np.mean(pnl_list)
    dd0 = np.mean(dd_list)
    trade0 = np.mean(num_trade_list)

    print( pnl0, dd0, trade0 )
    print( pnl_mat )
    print( dd_mat )

    #---------------- interpolate result in 2d -----------------------------------------
    interp_model_pnl = interpolate.interp2d( upper, lower, pnl_mat, kind=interp_kind )
    interp_model_dd = interpolate.interp2d( upper, lower, dd_mat, kind=interp_kind )
    interp_model_trade = interpolate.interp2d( upper, lower, trade_mat, kind=interp_kind )

    # define xy axis with more points
    y = np.linspace( min(upper), max(upper), len(upper)*8 )
    x = np.linspace( min(lower), max(lower), len(lower)*8 )

    # apply interpolation
    pnl_mat_interp = interp_model_pnl(y,x)
    dd_mat_interp = interp_model_dd(y,x)
    trade_mat_interp = interp_model_trade(y,x)

    # saving the parameter in txt file
    max_ind = np.unravel_index( pnl_mat_interp.argmax(), pnl_mat_interp.shape)
    upper_max = y[ max_ind[0] ]
    lower_max = x[ max_ind[1] ]
    max_pnl_str = np.max(pnl_mat_interp)
    max_dd_str = dd_mat_interp[ max_ind[0], max_ind[1] ]
    max_trade_str = trade_mat_interp[ max_ind[0], max_ind[1] ]
    final = ( new_model['tech'],  'upper= ' + str(upper_max), 'lower= ' + str(lower_max), 'max_pnl='+str(max_pnl_str), 'pnl_original='+str(pnl0), 'currency= '+cur, 
              'agent_tech= '+agent_tech, 'max_dd='+str(max_dd_str), 'dd0='+str(dd0), 'max_trade='+str(max_trade_str), 'trade0='+str(trade0) )
    f = open('../result.txt', 'w')
    f.write(" ".join(final) )
    f.close()

    # save the result image (cum pnl)
    plt.imshow( pnl_mat_interp, extent=[min(lower), max(lower), max(upper), min(upper)] )
    plt.colorbar(); plt.xlabel('lower'); plt.ylabel('upper')
    plt.savefig('../res_pnl.png')
    #plt.show()      
    plt.close()

    # save the result image (max dd)
    #plt.imshow( dd_mat_interp, extent=[min(lower), max(lower), max(upper), min(upper)] )
    #plt.colorbar(); plt.xlabel('lower'); plt.ylabel('upper')
    #plt.savefig('../res_dd.png')
    #plt.show()  
    #plt.close()
    
            
del_logs()

#---------------- 2nd run for pnl matrix ------------------------------------------------

if first_run==True and third_run==True:
    num_run = 2
elif first_run==False and third_run==False and test_model==True:
    num_run = 0
else:
    num_run = 1
    
if first_run==False and third_run==False and test_model==False:
    print("Everything set false.  Exiting.")
    exit()
    
for n in range(num_run):

    if third_run==True and first_run==False:
        os.chdir(output_dir)
        res2_file = open("../result2.txt", "r")
        res2_list = res2_file.read().split(" ")
        upper_max = float(res2_list[2])
        lower_max = float(res2_list[4])
        upper2 = np.linspace( upper_max+0.2, upper_max-0.2, 4 )     
        lower2 = np.linspace( lower_max+0.2, lower_max-0.2, 4 )     
    elif first_run==True and third_run==True and n==0:
        upper2 = np.linspace( upper_max+0.25, upper_max-0.25, 4 )     
        lower2 = np.linspace( lower_max+0.25, lower_max-0.25, 4 )  
    elif first_run==True and third_run==True and n==1:
        upper2 = np.linspace( upper_max+0.2, upper_max-0.2, 4 )     
        lower2 = np.linspace( lower_max+0.2, lower_max-0.2, 4 )     
        
    pnl_mat = np.zeros([len(upper2),len(lower2)])
    dd_mat = np.zeros([len(upper2),len(lower2)])
    trade_mat = np.zeros([len(upper2),len(lower2)])

    total_run=0
    for i, x1 in enumerate(upper2):
        for j, x2 in enumerate(lower2):
            if x1>x2 and (x1-x2)>=1.0 and (x1-x2)<=2.5:
                total_run+=1
                print( x1, x2 )
    
    l = 0
    for i, x1 in enumerate(upper2):
        for j, x2 in enumerate(lower2):
            
            if x1>x2 and (x1-x2)>=1.0 and (x1-x2)<=2.5:
            
                #x1 = 0.8337882547559966; x2 = -1.2174937965260546;
                new_model = { "tech":technical, "upper":x1, "lower":x2 }
                l += 1
                print(); print(" *************** param2 = " + str(l) + " of " + str(total_run) ); print()
                
                os.chdir(core_dir)
                ray.init()
                time.sleep(2)
                print(" check1" )
                ray0 = fun0.remote(new_model)
                time.sleep(2)
                print(" check2" )
                ray1 = fun1.remote(new_model)
                time.sleep(2)
                print(" check3" )
                ray2 = fun2.remote(new_model)
                time.sleep(2)
                print(" check4" )
                ray3 = fun3.remote(new_model)
                time.sleep(2)
                print(" check5" )
                ray4 = fun4.remote(new_model)
                print(" chech x " )
                time.sleep(2)
                ray_res = ray.get([ray0,ray1,ray2,ray3,ray4])
                ray.shutdown()
                del_logs()
                gc.collect()
                print()
                
                pnl_list, dd_list, num_trade_list = get_pnls()
                                        
                pnl_mat[i,j] = np.mean(pnl_list)
                dd_mat[i,j] = np.mean(dd_list)
                trade_mat[i,j] = np.mean(num_trade_list)
                
                print( pnl_mat )
                print( dd_mat )
                
                # create log
                final = " param = " + str(l) + " of " + str(total_run)
                if not os.path.exists("../log"):
                    os.makedirs("../log")
                if first_run==True and n==0:
                    log_dir = "../log/param2_" + str(l) + "of" + str(total_run) + ".txt"
                elif first_run==True and third_run==True and n==1:
                    log_dir = "../log/param3_" + str(l) + "of" + str(total_run) + ".txt"
                elif first_run==False and third_run==True and n==0:
                    log_dir = "../log/param3_" + str(l) + "of" + str(total_run) + ".txt"
                f = open(log_dir, 'w')
                f.write( final )
                f.close()


    #---------------- 2nd interpolation  ---------------------------------------------
    interp_model_pnl = interpolate.interp2d( upper2, lower2, pnl_mat, kind=interp_kind )
    interp_model_dd = interpolate.interp2d( upper2, lower2, dd_mat, kind=interp_kind )
    interp_model_trade = interpolate.interp2d( upper2, lower2, trade_mat, kind=interp_kind )

    # define xy axis with more points
    y = np.linspace( min(upper2), max(upper2), len(upper2)*8 )
    x = np.linspace( min(lower2), max(lower2), len(lower2)*8 )

    # apply interpolation
    pnl_mat_interp = interp_model_pnl(y,x)
    dd_mat_interp = interp_model_dd(y,x)
    trade_mat_interp = interp_model_trade(y,x)

    # saving the parameter in txt file
    max_ind = np.unravel_index( pnl_mat_interp.argmax(), pnl_mat_interp.shape)
    upper_max = y[ max_ind[0] ]
    lower_max = x[ max_ind[1] ]
    max_pnl_str = np.max(pnl_mat_interp)
    max_dd_str = dd_mat_interp[ max_ind[0], max_ind[1] ]
    max_trade_str = trade_mat_interp[ max_ind[0], max_ind[1] ]
    final = ( new_model['tech'],  'upper= ' + str(upper_max), 'lower= ' + str(lower_max), 'max_pnl='+str(max_pnl_str), 'currency= '+cur,
                'agent_tech= '+str(agent_tech),'max_dd='+str(max_dd_str), 'max_trade='+str(max_trade_str) )
    if first_run==True and n==0:
        f = open('../result2.txt', 'w')
    elif first_run==True and third_run==True and n==1:
        f = open('../result3.txt', 'w')
    elif first_run==False and third_run==True and n==0:
        f = open('../result3.txt', 'w')
    f.write(" ".join(final) )
    f.close()

    # save the result image pnl
    plt.imshow( pnl_mat_interp, extent=[min(lower2), max(lower2), max(upper2), min(upper2)] )
    plt.colorbar(); plt.xlabel('lower'); plt.ylabel('upper')
    if first_run==True and n==0:
        plt.savefig('../res2_pnl.png')
    elif first_run==True and third_run==True and n==1:
        plt.savefig('../res3_pnl.png')
    elif first_run==False and third_run==True and n==0:
        plt.savefig('../res3_pnl.png')
    plt.close()
    
    # save the result image dd
    #plt.imshow( dd_mat_interp, extent=[min(lower2), max(lower2), max(upper2), min(upper2)] )
    #plt.colorbar(); plt.xlabel('lower'); plt.ylabel('upper')
    #if first_run==True and n==0:
    #    plt.savefig('../res2_dd.png')
    #elif first_run==True and third_run==True and n==1:
    #    plt.savefig('../res3_dd.png')
    #elif first_run==False and third_run==True and n==0:
    #    plt.savefig('../res3_dd.png')
    #plt.close()

del_logs()
print("Done!!")
          
          
#-----------------------  test model ------------------------------------------      
            
if test_model==True and first_run==False and third_run==False:

    new_model = { "tech":technical, "upper":test_upper, "lower":test_lower }

    print("testing model once... ")
    os.chdir(core_dir)
    ray.init()
    time.sleep(2)
    print(" check1" )
    ray0 = fun0.remote(new_model)
    time.sleep(2)
    print(" check2" )
    ray1 = fun1.remote(new_model)
    time.sleep(2)
    print(" check3" )
    ray2 = fun2.remote(new_model)
    time.sleep(2)
    print(" check4" )
    ray3 = fun3.remote(new_model)
    time.sleep(2)
    print(" check5" )
    ray4 = fun4.remote(new_model)
    print(" chech x " )
    time.sleep(2)
    ray_res = ray.get([ray0,ray1,ray2,ray3,ray4])
    ray.shutdown()
    del_logs()
    gc.collect()
    print()

    pnl_list, dd_list, num_trade_list = get_pnls()

    pnl0 = np.mean(pnl_list)
    dd0 = np.mean(dd_list)
    trade0 = np.mean(num_trade_list)

    print("pnl = ", pnl0, " dd = ", dd0, " num_trade = ", trade0 )

    
    
    
    
    
            
