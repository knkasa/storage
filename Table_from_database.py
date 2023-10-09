import sys
import os
import uuid
import json, sys
import sqlalchemy as sa
import pandas as pd
import pickle
from pandasql import sqldf

# Note:  https://jira.tyo.invastsec.com/browse/MS-1859   If you need to check if the user is removed from account, then the agent will be null

mysql_engine = sa.create_engine('postgresql+psycopg2://xxx:yyy@zzz.com/ai_db')     # new prod 2.0 


#str_query_agent_ai_decisions = " SELECT * FROM INFORMATION_SCHEMA.TABLES "      # this will list all schemas. 
str_query_agent_ai_decisions = " SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA    "   # this will list all schemas.  select * from <schema>.<table_name> where table_type='BASE_TABLE' 

agent_id =  '8bfbad80-3a26-49b4-9669-10a4de0feabe'
agent_id2 = 'ai-36bf5d6dbf723d17c0237d66b0baa9a65b51441ed52e605e65fe60d493dae3de-1592092407579'

print( mysql_engine )
print("List of schemas = ",  mysql_engine.table_names() )

str_query_agent_ai_decisions = " select * from   agent_ai_decisions  where  agent_id=127257    order by utc_datetime "
#str_query_agent_ai_decisions = " select * from   agent_ai_decisions     order by utc_datetime  limit 50000 "
#str_query_agent_ai_decisions = " select * from   agent_ai_decisions  where  sim_type='daily_signal'   order by utc_datetime   "
#str_query_agent_ai_decisions = " select * from agent_ai   "

#str_query_agent_ai_decisions = " select * from learning_price_tech_indices where utc_datetime>='2020-06-01' and utc_datetime<='2020-07-01' and instrument='USDJPY' order by utc_datetime asc limit 10000000 "  

#str_query_agent_ai_decisions = "select * from agent_ai_decisions where agent_id='" + agent_id + "' and  sim_type='daily_signal'  order by utc_datetime"
#str_query_agent_ai_decisions = "select * from agent_ai_decisions where agent_id='" + agent_id + "'  order by utc_datetime"
#str_query_agent_ai_decisions = " select * from agent_ai_decisions where agent_id in  ('" + agent_id + "' , '" + agent_id2  + "')  and  sim_type='daily_signal'  order by utc_datetime "
#str_query_agent_ai_decisions = "select * from agent_ai_decisions where agent_id in  ('" + agent_id + "' , '" + agent_id2  + "')  order by utc_datetime"

#str_query_agent_ai_decisions = "select  *  from agent_ai_decisions where agent_id='" + agent_id + "' and  utc_datetime>='2021-06-01' order by utc_datetime  "

#str_query_agent_ai_decisions = "select a.agent_id, a.utc_datetime, a.realized_pnl, b.agent_id, b.original_agent_id, b.instrument, b.technical_index from agent_ai_decisions as a, agent_ai as b " + \
#                                 " where a.agent_id=b.agent_id and  a.agent_id>=1011 and a.agent_id<=1020  order by a.utc_datetime "


#str_query_agent_ai_decisions = "select  *   from   agent_ai_decisions where sim_type='daily_signal'   order by utc_datetime   "

# join
#select ss.*, aa.instrument, aa.agent_appetite, aa.technical_index, aa.news_type from signal_stats ss 
#        left join agent_ai aa on aa.agent_id = ss.agent_id where  ss.past_days = 7 and ss.agent_id in (

# multipe tables
# str_query = "select a.*, b.total_signals from signal_stats_ranking as a, signal_stats as b where a.agent_id=b.agent_id and a.base_date=b.base_date and b.past_days=365  "


#str_query_agent_ai_decisions = " select * from symbol_swap  limit 10000 "

#str_query_agent_ai_decisions = " select  *   from   price_history_candlestick  where instrument='ZARJPY'  order by  utc_datetime desc  limit 10000 "  # BTCJPY,  BTCUSD
#str_query_agent_ai_decisions = " select  *   from   price_history_candlestick  where instrument='ETHJPY' order by utc_datetime desc limit 10000 "  # BTCJPY,  BTCUSD
#str_query_agent_ai_decisions = " select  *   from   price_history_candlestick  where instrument='XRPUSD'  order by utc_datetime asc  limit 10000 "  # BTCJPY,  BTCUSD
#str_query_agent_ai_decisions = "select  *  from price_history_candlestick where instrument='XRPUSD' and utc_datetime BETWEEN '2021-11-15 11:00:00' and '2022-01-23 11:00:00' order by utc_datetime "
#str_query_agent_ai_decisions = " select  *   from   price_history_candlestick  where instrument='ZARJPY' and utc_datetime>='2023-04-20' and utc_datetime<='2023-05-10'  order by  utc_datetime asc "    # BTCJPY,  BTCUSD

#str_query_agent_ai_decisions = " select  *   from   price_history_candlestick where instrument='SPCUSD'  order by  utc_datetime desc  limit 10000 "  #  CFD query


#str_query_agent_ai_decisions = "select utc_datetime, assetcode, datatype, ( COALESCE(+surprise, 0) + COALESCE(-timeUrgency, 0) + COALESCE(-uncertainty, 0) + COALESCE(-emotionVsFact, 0) ) * buzz as TRMI_CUR_tone  from currency_news  where utc_datetime>'2022-08-01' and datatype='News_Social' and assetcode='JPY' "

#str_query_agent_ai_decisions = " select  *  from   signal_stats_ranking  where base_date>='2022-02-01'   "
#str_query_agent_ai_decisions = " select  agent_id, max(episode), sum(realized_pnl) as sum_realized, count(realized_pnl) as count_realized  from agent_ai_decisions where utc_datetime>='2021-07-01' and realized_pnl<>0 group by agent_id  order by agent_id  " 

#str_query_agent_ai_decisions = " select agent_id,  sum(realized_pnl) as sum_pnl, count(realized_pnl) as num_trade from   agent_ai_decisions where  utc_datetime>='2021-01-01' and realized_pnl<>0.0  group by agent_id order by num_trade  "
#str_query_agent_ai_decisions = " select agent_id,  sum(realized_pnl) as sum_pnl, count(realized_pnl) as num_trade from   agent_ai_decisions    group by agent_id order by num_trade  "


#str_query_agent_ai_decisions = " select  *  from   agent_ai_decisions  where   utc_datetime>='2017-10-26' and utc_datetime<'2017-10-30'  order by utc_datetime  "

#str_query_agent_ai_decisions = " select  utc_datetime, buzz, surprise, timeUrgency, uncertainty, emotionVsFact  from currency_news where utc_datetime>'2022-09-01' and datatype='News_Social' and assetcode='JPY'  order by utc_datetime  "
#str_query_agent_ai_decisions = "select utc_datetime, assetcode, datatype, ( COALESCE(-surprise, 0)  ) * 1 as TRMI_CUR_tone  from currency_news  where utc_datetime>'2022-10-01' and datatype='News_Social' and assetcode='JPY' "
#str_query_agent_ai_decisions = " select *  from market_news where   datatype='News_Social' and assetcode='US' order by utc_datetime limit 10000  "  

str_query_agent_ai_decisions = " select * from learning_news_indices limit 10000 " 
#str_query_agent_ai_decisions = " select * from mm_trading.symbols   order by business_day   "   # monolith swap data

#------------------------------------------------------------------------------------------------------------------------------------------------

#str_query_agent_ai_decisions = "select agent_id, utc_datetime, episode from agent_ai_decisions where utc_datetime>='2021-09-16'  and episode<3000 order by episode "

#str_query_agent_ai_decisions = "select * from signal_stats_ranking  where agent_id='" + agent_id + "' "

#str_query_agent_ai_decisions = " SELECT agent_id, episode, COUNT(episode) AS episodes_count, MIN(utc_datetime) AS from_date, max(utc_datetime) AS to_date_ FROM agent_ai_decisions WHERE utc_datetime>'2021-05-01' and sim_type='daily_signal' group BY agent_id, episode having COUNT(episode)>5 ORDER BY from_date desc LIMIT 300"

#str_query_agent_ai_decisions = "select  agent_id, instrument, technical_index, utc_datetime, episode from agent_ai_decisions where utc_datetime>='2021-06-03 00:00:00'  "

#str_query_agent_ai_decisions = "select * from agent_ai_decisions where  sim_type='daily_signal'  order by utc_datetime  limit 1000"
#str_query_agent_ai_decisions = "select  agent_id, instrument, technical_index, news_type, agent_appetite   from   agent_ai "

#str_query_agent_ai_decisions = "select  *  from signal_stats where     base_date>='2021-09-20 00:00:00'  order by agent_id "
#str_query_agent_ai_decisions = "select  *  from signal_stats where   base_date between '2021-04-21 00:00:00' and '2021-04-21 23:59:00'  order by agent_id "

#str_query_agent_ai_decisions = "SELECT datetime_open, datetime_close, price_open, price_low, price_high, price_close FROM (SELECT max(bid_high) AS price_high, min(bid_low) AS price_low from price_history_candlestick where instrument = 'USDJPY' and utc_datetime BETWEEN '2021-10-19 11:00:00+00:00' and '2021-10-19 09:47:00+00:00' GROUP BY instrument) AS ST,   (SELECT utc_datetime AS datetime_open, bid_open AS price_open FROM price_history_candlestick where instrument = 'USDJPY' and utc_datetime >= '2021-05-21 11:09:00+00:00'  ORDER BY utc_datetime ASC LIMIT 1) AS OP,  (SELECT utc_datetime AS datetime_close, bid_close AS price_close FROM price_history_candlestick where instrument = 'USDJPY' and utc_datetime = '2020-09-11 09:47:00+00:00') AS CL "
#str_query_agent_ai_decisions = "select  *  from signal_stats where total_signals>=40 and past_days=365 and   base_date   between '2020-11-30 00:00:00' and '2020-11-30 23:59:00'  order by agent_id "

#str_query_agent_ai_decisions = "select  distinct(agent_id), instrument,  agent_appetite,  technical_index,  news_type  from agent_ai where latest_utc_datetime>='2021-03-31 00:00:00' "

#str_query_agent_ai_decisions = "select  *  from  signal_stats_ranking  where  base_date  between '2020-11-30 00:00:00' and '2020-12-01 23:59:00' "

#str_query_agent_ai_decisions = "select agent_decision_id, agent_id, instrument, utc_datetime, ordered_price, realized_pnl, unrealized_pnl, sim_type, bid_price_close, ask_price_close, trade_status from agent_ai_decisions where utc_datetime>= '2020-04-10 05:48:08'order by utc_datetime"
#str_query_agent_ai_decisions = "select * from agent_ai_decisions where utc_datetime>= '2020-04-10 05:48:08'order by utc_datetime"

#str_query_agent_ai_decisions = "select agent_id, utc_datetime, realized_pnl, unrealized_pnl, current_position, bid_price_close from agent_ai_decisions where utc_datetime between  '2021-02-20 00:00:00' and '2021-03-31 23:59:00'  and instrument='GBPUSD' and sim_type='daily_signal' " 

#str_query_agent_ai_decisions = "select agent_id, utc_datetime, realized_pnl, unrealized_pnl, current_position from   agent_ai_decisions  where instrument='GBPUSD' and sim_type='daily_signal' and  utc_datetime between '2020-06-01 00:00:00' and '2021-01-31 23:59:00' order by utc_datetime "

#str_query_agent_ai_decisions = " select x.agent_id from (select agent_id, sum(realized_pnl) as sum_pnl from agent_ai_decisions where sim_type='daily_signal'  group by agent_id) as x where x.sum_pnl=0.0  "





#str_query = "select agent_id, instrument, date_trunc('day', utc_datetime) as yyyymmdd, trade_status, sum(realized_pnl) as realized_pnl from agent_ai_decisions " + \
#            "where sim_type='daily_signal' group by agent_id, instrument, date_trunc('day', utc_datetime), trade_status order by agent_id, date_trunc('day', utc_datetime)   "

"""select distinct ss.symbol as instrument,
        date_trunc('day', ss.original_signal_time) as base_date,
        sum(coalesce(ss.realized_pnl_pips, 0)) over 
        (partition by date_trunc('day', ss.original_signal_time), ss.symbol) as sum_realized,
        sum(coalesce(ss.unrealized_pnl_pips, 0)) over 
        (partition by date_trunc('day', ss.original_signal_time), ss.symbol) as sum_unrealized,
        count(coalesce(ss.source_id, 0)) over
        (partition by date_trunc('day', ss.original_signal_time), ss.symbol) as cnt_agents,
        from sms_signals as ss
        where ss.original_signal_time between %(start_day)s::timestamp and %(to_day)s::timestamp
        and ss.realized_pnl_pips != 0
        order by base_date, instrument
        """

'''
with observers as (
	select distinct x.source_id
	from (
		select ss2.source_id,
		sum(coalesce(ss2.realized_pnl_pips, 0)) as sum_realized_pnl
		from mm_agent.sms_signals ss2 
		join mm_agent.agents a2 on (ss2.source_id = a2.agent_id)
		where ss2.original_signal_time between '2021-01-01 00:00:00'::timestamp and '2021-12-06 23:59:59'::timestamp
		and ss2.original_signal_time  >= a2.activated_at 
		group by ss2.source_id 
	) as x where x.sum_realized_pnl::numeric(10,1) = 0::integer
)
select distinct ss.symbol as instrument,
date_trunc('day', ss.original_signal_time) as base_date,
sum(coalesce(ss.realized_pnl_pips, 0)) over 
(partition by date_trunc('day', ss.original_signal_time), ss.symbol) as sum_realized,
sum(coalesce(ss.unrealized_pnl_pips, 0)) over 
(partition by date_trunc('day', ss.original_signal_time), ss.symbol) as sum_unrealized,
count(coalesce(ss.source_id, 0)) over
(partition by date_trunc('day', ss.original_signal_time), ss.symbol) as cnt_agents
from sms_signals as ss
join agents a on (ss.source_id = a.agent_id)
where ss.original_signal_time between '2021-01-01 00:00:00'::timestamp and '2021-12-06 23:59:59'::timestamp
and ss.source_id not in (select source_id from observers)
and ss.original_signal_time >= a.activated_at
and a.owner_user_id is not NULL
order by instrument, base_date;
'''

'''
with test1 as(
select uriage from tokuisaki
where uriage > 100
)
select b.tokuisaki, b.uriage, b.YYYYMM
from test1 a,tokuisaki b
where b.uriage in (a.uriage);
'''



pdf_agent_ai_decisions = pd.read_sql_query(str_query_agent_ai_decisions, mysql_engine)
pdf_agent_ai_decisions.reset_index(drop=True, inplace=True)
print("Table = ", pdf_agent_ai_decisions  )
#import pdb; pdb.set_trace()  

#print( pdf_agent_ai_decisions.columns )
print("Table info = ",  pdf_agent_ai_decisions.info()  )

pdf_agent_ai_decisions.to_csv('C:/my_working_env/download_checkTable/data.csv', index=None)
import pdb; pdb.set_trace()

