import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import csv
import random

import sqlalchemy as sa
from jinjasql import JinjaSql
from copy import deepcopy
from six import string_types

#https://towardsdatascience.com/a-simple-approach-to-templated-sql-queries-in-python-adc4f0dc511
#https://towardsdatascience.com/advanced-sql-templates-in-python-with-jinjasql-b996eadd761d

mysql_engine = sa.create_engine('postgresql+psycopg2://inv-ai-role-adm-stg:jPCh22G8@mm-ai-db-ro.mm-stg.invastsec.com/ai_db') 

def quote_sql_string(value):
    # put ' quotation inside where statement
    if isinstance(value, string_types):
        new_value = str(value)
        new_value = new_value.replace("'", "''")
        return "'{}'".format(new_value)
    return value

def get_sql_from_template(query, bind_params):
    if not bind_params:
        return query
    params = deepcopy(bind_params)
    for key, val in params.items():
        params[key] = quote_sql_string(val)  # put ' quotation to where statement 
    return query % params  # insert params into query
    
def apply_sql_template(template, params):
    j = JinjaSql(param_style='pyformat')
    query, bind_params = j.prepare_query(template, params)
    return get_sql_from_template(query, bind_params)
    
def multi_cols(template, params):
    if isinstance(params['cols'], list):
        params['cols'] = ', '.join(params['cols'])
    return apply_sql_template(template, params)
    
#------------- example 1 -------------------------------------------------
# sqlsafe removes quotation mark.
template = ''' select {{col_name | sqlsafe}}  from {{table | sqlsafe}}
   where agent_id={{agent_id}} order by utc_datetime '''
               
params = { 'table': 'agent_ai_decisions',
            'col_name': ['utc_datetime'],
            'agent_id': '1b0839c1-0082-4966-994f-1ced29925045' }
            
sql_q = apply_sql_template(template, params)
print(); print(sql_q)


#-------------- example 2 -------------------------------------------------
template = '''  {% set col_name = ', '.join(cols) %}
    select {{col_name | sqlsafe}} from {{table | sqlsafe}}
    where agent_id in {{ ids | inclause}} order by utc_datetime asc'''
               
params = { 'table': 'agent_ai_decisions',
            'cols': ['utc_datetime', 'agent_id', 'realized_pnl'],
            'ids': ['1b0839c1-0082-4966-994f-1ced29925045', 'ai-514d7aee23a2-1562570801679'] }
            
sql_q = apply_sql_template(template, params)
print(); print(sql_q)


#---------------- example 3 ------------------------------------------------
template = '''  select {{ cols | sqlsafe }} from {{ table | sqlsafe }}
    where agent_id in {{ ids | inclause }} order by utc_datetime asc  '''

params = { 'table': 'agent_ai_decisions',
            'cols': ['utc_datetime', 'agent_id', 'realized_pnl'],  # or you could simply 'x1, x2' as one string
            'ids': ['1b0839c1-0082-4966-994f-1ced29925045', 'ai-514d7aee23a2-1562570801679'] }
       
sql_q = multi_cols( template, params )
print(); print(  sql_q   )

table = pd.read_sql_query(sql_q, mysql_engine)
#print( table )


#---------------- test ------------------------------------------------------
template = '''  select {{ cols | sqlsafe }} from {{ table | sqlsafe }}
    where agent_id in {{ ids | inclause }} and sim_type='daily_signal' 
    order by utc_datetime asc  '''


os.chdir('C:/my_working_env/download_checkTable')
with open('./sql_list.csv', 'r',  encoding="utf-8") as ff:   #******only change file name here
	xx = csv.reader(ff)
	xlist = []
	for x in xx:
		xlist.extend(x)
         
#random.shuffle(xlist)
#xlist = xlist[0:1000]

params = { 'table': 'agent_ai_decisions',
            'cols': '*',  # ['utc_datetime', 'agent_id', 'realized_pnl'],  # or you could simply 'x1, x2' as one string
            'ids': xlist  }
       
sql_q = multi_cols( template, params )
table = pd.read_sql_query(sql_q, mysql_engine)
print( table )

table.to_csv('C:/Users/ken_nakatsukasa/Desktop/download_checkTable/data.csv')






