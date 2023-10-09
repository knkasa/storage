
import pandas as pd
import numpy as np
import os
import gc
import re
import pdb
import matplotlib.pyplot as plt
import japanize_matplotlib
from pandasql import sqldf
from matplotlib.pyplot import figure
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.font_manager as fm
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import optuna
import warnings
#warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
fprop = fm.FontProperties(fname='C:/Users/knkas/Desktop/ey_case_study/japanese_font/static/NotoSansJP-Black.ttf')
from mpl_toolkits.basemap import Basemap


os.chdir('C:/Users/knkas/Desktop/ey_case_study')


df_customer = pd.read_csv('./data/olist_customers_dataset.csv')
df_location = pd.read_csv('./data/olist_geolocation_dataset.csv')
df_item = pd.read_csv('./data/olist_order_items_dataset.csv')
df_payment = pd.read_csv('./data/olist_order_payments_dataset.csv')
df_review = pd.read_csv('./data/olist_order_reviews_dataset.csv')
df_order_status = pd.read_csv('./data/olist_orders_dataset.csv')
df_product = pd.read_csv('./data/olist_products_dataset.csv')
df_name = pd.read_csv('./data/product_category_name_translation.csv')
df_seller = pd.read_csv('./data/olist_sellers_dataset.csv')

df_customer = pd.merge( df_customer, df_location, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='inner' )
df_customer.drop_duplicates(subset =['customer_id'], keep="first", ignore_index=True, inplace=True)
df_customer.drop( columns=['geolocation_zip_code_prefix', 'geolocation_city', 'geolocation_state'], axis=1, inplace=True )
df_customer.rename( columns={'geolocation_lat': 'customer_lat', 'geolocation_lng':'customer_lng'}, inplace=True )
df_seller = pd.merge( df_seller, df_location, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='inner' )
df_seller.drop_duplicates(subset =['seller_id'], keep="first", ignore_index=True, inplace=True)
df_seller.drop( columns=['geolocation_zip_code_prefix', 'geolocation_city', 'geolocation_state'], axis=1, inplace=True )
df_seller.rename( columns={'geolocation_lat': 'seller_lat', 'geolocation_lng':'seller_lng'}, inplace=True ) 
gc.collect()

# change data format
#df_customer['customer_zip_code_prefix'] = df_customer['customer_zip_code_prefix'].astype(str)
#df_location['geolocation_zip_code_prefix'] = df_location['geolocation_zip_code_prefix'].astype(str)
#df_item['order_item_id'] = df_item['order_item_id'].astype(str)
df_review['review_score'] = df_review['review_score'].astype(int)
#df_item['shipping_limit_date'] = pd.to_datetime( df_item.shipping_limit_date )
#df_review['review_creation_date'] = pd.to_datetime( df_review.review_creation_date )
#df_review['review_answer_timestamp'] = pd.to_datetime( df_review.review_answer_timestamp )
df_order_status['order_purchase_timestamp'] = pd.to_datetime( df_order_status.order_purchase_timestamp )
#df_order_status['order_approved_at'] = pd.to_datetime( df_order_status.order_approved_at )
df_order_status['order_delivered_carrier_date'] = pd.to_datetime( df_order_status.order_delivered_carrier_date )
df_order_status['order_delivered_customer_date'] = pd.to_datetime( df_order_status.order_delivered_customer_date )
df_order_status['order_estimated_delivery_date'] = pd.to_datetime( df_order_status.order_estimated_delivery_date )

df_customer.drop( columns=['customer_zip_code_prefix'], axis=1, inplace=True )
df_location.drop( columns=['geolocation_zip_code_prefix'], axis=1, inplace=True )
df_item.drop( columns=['order_item_id', 'shipping_limit_date'], axis=1, inplace=True )
df_payment.drop( columns=['payment_sequential'], axis=1, inplace=True )
df_review.drop( columns=['review_creation_date','review_answer_timestamp'], axis=1, inplace=True )
df_order_status.drop( columns=['order_approved_at'], axis=1, inplace=True )
df_product.drop( columns=['product_name_lenght','product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm' ], axis=1, inplace=True )
df_seller.drop( columns=['seller_zip_code_prefix',], axis=1, inplace=True ) 
gc.collect()

#df_location = df_location.rename( columns={'geolocation_state': 'customer_state'} )
df_payment = pd.merge(df_payment, df_item, on='order_id', how='left')
df_payment = pd.merge(df_payment, df_review, on='order_id', how='left')
df_payment = pd.merge(df_payment, df_order_status, on='order_id', how='left')
df_payment = pd.merge(df_payment, df_customer, on='customer_id', how='left')
df_payment = pd.merge(df_payment, df_product , on='product_id', how='left')
df_payment = pd.merge(df_payment, df_name , on='product_category_name', how='left')
df_payment = pd.merge(df_payment, df_seller , on='seller_id', how='left')
df_payment.drop( columns=['product_category_name'], axis=1, inplace=True )
df_payment.drop( columns=['review_id'], axis=1, inplace=True )

df_payment['deliver_days'] = (df_payment['order_delivered_customer_date'] - df_payment['order_purchase_timestamp']).dt.days
df_payment['expected_deliver_diff'] = (df_payment['order_estimated_delivery_date'] - df_payment['order_delivered_customer_date']).dt.days
df_payment['order_shipped_days'] = (df_payment['order_delivered_carrier_date'] - df_payment['order_purchase_timestamp']).dt.days
df_payment['shipping_days'] = df_payment['deliver_days'] - df_payment['order_shipped_days']

df_payment.dropna(subset=['order_delivered_customer_date'], inplace=True )
df_payment.sort_values('order_delivered_customer_date', inplace=True)
df_payment.drop_duplicates(subset =['order_id','order_purchase_timestamp'], keep="first", ignore_index=True, inplace=True)

def calculate_distance(x):
    lat_diff = np.power(x.customer_lat - x.seller_lat, 2)
    lng_diff = np.power(x.customer_lng - x.seller_lng, 2)
    return np.sqrt( lat_diff + lng_diff ) 

df_payment['distance'] = df_payment.apply( lambda x: calculate_distance(x), axis=1 )
df_payment.drop( columns=['customer_lat','customer_lng','seller_lat','seller_lng'], axis=1, inplace=True )

del df_item
del df_review
#del df_order_status
del df_customer
del df_product
del df_name
del df_seller
gc.collect()

pdb.set_trace()
#-------- review score pie chart ----------------------------------------------
df_payment['review_score_str'] = df_payment["review_score"].fillna( value="空白" ) 
value_counts = df_payment["review_score_str"].value_counts(dropna=False)
colors = ['#FBB4AE', '#B3CDE3', '#CCEBC5', '#DECBE4', '#FED9A6', '#FFFFCC']
plt.pie( value_counts, labels=value_counts.index.tolist(), autopct='%1.1f%%', colors=colors, startangle=50)
plt.axis('equal')
plt.title("レビュースコア", fontsize=16)
plt.savefig('review.png' ,  dpi=200 )
#plt.show()
plt.close()

#-------- order status pie chart ----------------------------------------------
df_order_status['order_status'] = df_order_status["order_status"].fillna( value="空白" ) 
value_counts = df_order_status["order_status"].value_counts(dropna=False)
#cmap = plt.get_cmap('tab10')
#colors = cmap(np.linspace(0, 1, len(value_counts))) 
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd', '#9a9c9a', '#ed8549']
fig, ax = plt.subplots()
wedges, _, _ = ax.pie( 
                    value_counts,    
                    labels=None, 
                    colors=colors, 
                    autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else '' 
                    )
ax.legend(wedges, value_counts.index.tolist(), loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_aspect('equal')
ax.set_title('オーダーステータス', fontsize=16)
plt.tight_layout()
plt.savefig('order.png' ,  dpi=200 )
#plt.show()
plt.close()

#------- total transaction monthly ---------------------------------------------
monthly_payment = df_payment.groupby( pd.Grouper(key='order_delivered_customer_date', freq='M') )['payment_value'].count()
figure(num=None, figsize=(7, 4), dpi=100, facecolor='w', edgecolor='k') 
plt.plot(monthly_payment.index, monthly_payment.values, color='blue', linestyle='-', marker='o')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(['取引回数'], loc='upper left', )
#plt.xlabel('Year-Month')
plt.ylabel('取引回数', )
plt.title('取引回数（月次）', fontsize=16, )
plt.xticks(rotation=25)
plt.savefig('num_delivered_vs_time.png' ,  dpi=200 )
#plt.show()
plt.close()

#monthly_payment.to_csv('monthly_payment.csv', index=None)

#--------- new customer monthly -------------------------------------------

'''
# Create a set to store unique customer_ids
unique_customers = set()

# Initialize an empty list to store the count of new customer_ids for each month
new_customer_counts = []

# Iterate over the DataFrame rows
for index, row in df_payment.iterrows():
    customer_id = row['customer_unique_id']
    order_date = row['order_delivered_customer_date']
    
    # Check if customer_id appeared in previous months
    if customer_id not in unique_customers:
        # Add customer_id to the set
        unique_customers.add(customer_id)
        
        # Count the number of new customer_ids for each month
        new_customer_counts.append((order_date.year, order_date.month, len(unique_customers)))

# Create a new DataFrame with the new customer counts
df_new_customers = pd.DataFrame(new_customer_counts, columns=['Year', 'Month', 'new_customer_count'])

df_new_customers['year_month'] = df_new_customers.apply( lambda x: str(x.Year) + '-' +  str(x.Month) , axis=1 )
df_new_customers = df_new_customers.groupby(['year_month']).count()

df_new_customers.index = pd.to_datetime(df_new_customers.index)
df_new_customers = df_new_customers.sort_index()

figure(num=None, figsize=(7, 4), dpi=100, facecolor='w', edgecolor='k') 
plt.plot( df_new_customers.index, df_new_customers.new_customer_count, color='blue', linestyle='-', marker='o' ) 
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(['購入数'], loc='upper left')
plt.xlabel('Year-Month')
plt.ylabel('購入数')
plt.title('新規の購入者数（月次）', fontsize=16 )
plt.xticks(rotation=25)
plt.savefig('new_customer_counts1.png' ,  dpi=200 )
plt.show()
plt.close()
'''

'''
# Create a set to store unique customer_ids
unique_customers = set()

# Initialize an empty list to store the count of new customer_ids for each month
new_customer_counts = []

# Iterate over the DataFrame rows
for index, row in df_payment.iterrows():
    customer_id = row['seller_id']
    order_date = row['order_delivered_customer_date']
    
    # Check if customer_id appeared in previous months
    if customer_id not in unique_customers:
        # Add customer_id to the set
        unique_customers.add(customer_id)
        
        # Count the number of new customer_ids for each month
        new_customer_counts.append((order_date.year, order_date.month, len(unique_customers)))

# Create a new DataFrame with the new customer counts
df_new_customers = pd.DataFrame(new_customer_counts, columns=['Year', 'Month', 'new_customer_count'])

df_new_customers['year_month'] = df_new_customers.apply( lambda x: str(x.Year) + '-' +  str(x.Month) , axis=1 )
df_new_customers = df_new_customers.groupby(['year_month']).count()

df_new_customers.index = pd.to_datetime(df_new_customers.index)
df_new_customers = df_new_customers.sort_index()

figure(num=None, figsize=(7, 4), dpi=100, facecolor='w', edgecolor='k') 
plt.plot( df_new_customers.index, df_new_customers.new_customer_count, color='blue', linestyle='-', marker='o' ) 
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(['販売数'], loc='upper left')
plt.xlabel('Year-Month')
plt.ylabel('販売数')
plt.title('新規の販売者数（月次）', fontsize=16 )
plt.xticks(rotation=25)
plt.savefig('new_customer_counts_sell.png' ,  dpi=200 )
plt.show()
plt.close()
'''

'''
# Create an empty DataFrame to store the counts
df_new_customers = pd.DataFrame(columns=['Year', 'Month', 'customer_unique_id'])

# Iterate over the DataFrame rows
n = 0
for index, row in df_payment.iterrows():
    n += 1
    print( n, len(df_payment) )
    customer_unique_id = row['customer_unique_id']
    order_date = row['order_delivered_customer_date']
    
    # Check if customer_unique_id appeared in previous months
    previous_customers = df_new_customers.loc[(df_new_customers['Year'] < order_date.year) |
                                              ((df_new_customers['Year'] == order_date.year) &
                                               (df_new_customers['Month'] < order_date.month)), 'customer_unique_id']
    
    if customer_unique_id not in previous_customers.values:
        # Append the count of new customer_ids for each month
        df_new_customers = df_new_customers.append({'Year': order_date.year, 'Month': order_date.month, 'customer_unique_id': customer_unique_id}, ignore_index=True)

# Calculate the number of new customers for each month
new_customer_counts = df_new_customers.groupby(['Year', 'Month']).size().reset_index(name='new_customer_count')
new_customer_counts.to_csv('new_customer_counts2.csv')
'''

#---------- top 10 category ---------------------------------------------------


# https://medium.com/@BrendanArtley/matplotlib-color-gradients-21374910584b
def hex_to_RGB(hex_str):
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]
def get_color_gradient(c1, c2, n):
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

category_counts = df_payment['product_category_name_english'].value_counts().sort_values(ascending=False).head(10)

#cmap = sns.color_palette('Blues_r', n_colors=len(category_counts))  # this is for gradient color.
sns.set_theme(style='darkgrid')

figure(num=None, figsize=(7, 4), dpi=100, facecolor='w', edgecolor='k') 
#plt.barh( category_counts.index, category_counts.values, color=cmap[::-1])  # use this if using sns library.
plt.barh( category_counts[::-1].index, category_counts[::-1].values, color=get_color_gradient("#8A5AC2", "#3575D5", len(category_counts)) )
plt.xlabel('Count')
#plt.xticks( fontproperties=fprop, )
#plt.ylabel('Product Category')
plt.tick_params(axis='y', length=0)
plt.title('購入数のカテゴリトップ10', fontproperties=fprop, fontsize=16 )
plt.tight_layout()
plt.savefig('top10_category_count.png' ,  dpi=200 )
#plt.show()
plt.close()

category_counts.to_csv('category_counts.csv', index=None)

#------------------- top 10 category in price --------------

category_price = df_payment.groupby(['product_category_name_english'])['price'].sum().sort_values(ascending=False).head(10)

#cmap = sns.color_palette('Blues_r', n_colors=len(category_price))   # this is for gradient color.
#sns.set_theme(style='darkgrid')

figure(num=None, figsize=(7, 4), dpi=100, facecolor='w', edgecolor='k') 
#plt.barh( category_price.index, category_price.values, color=cmap[::-1])  # use this if using sns library.
plt.barh( category_price[::-1].index, category_price[::-1].values, color=get_color_gradient("#8A5AC2", "#3575D5", len(category_price)) )
plt.xlabel('Count')
#plt.xticks( fontproperties=fprop, )
#plt.ylabel('Product Category')
plt.tick_params(axis='y', length=0)
plt.title('支払い総額の製品カテゴリトップ 10', fontproperties=fprop, fontsize=16 )
plt.tight_layout()
plt.savefig('top10_category_sum.png' ,  dpi=200 )
#plt.show()
plt.close()

#------------------- top 10 category in average price --------------

category_mean = df_payment.groupby(['product_category_name_english'])['price'].mean().sort_values(ascending=False).head(10)

#cmap = sns.color_palette('Blues_r', n_colors=len(category_mean))   # this is for gradient color.
#sns.set_theme(style='darkgrid')

figure(num=None, figsize=(7, 4), dpi=100, facecolor='w', edgecolor='k') 
#plt.barh( category_mean.index, category_mean.values, color=cmap[::-1])  # use this if using sns library.
plt.barh( category_mean[::-1].index, category_mean[::-1].values, color=get_color_gradient("#8A5AC2", "#3575D5", len(category_mean)) )
plt.xlabel('Count')
#plt.xticks( fontproperties=fprop, )
#plt.ylabel('Product Category')
plt.tick_params(axis='y', length=0)
plt.title('平均価格のカテゴリトップ 10', fontproperties=fprop, fontsize=16 )
plt.tight_layout()
plt.savefig('top10_category_price.png' ,  dpi=200 )
#plt.show()
plt.close()

#------------ geological map -----------------------------

'''
latitude = df_location.geolocation_lat
longitude = df_location.geolocation_lng

figure(num=None, figsize=(10, 5), dpi=200, facecolor='w', edgecolor='k') 
plt.scatter(longitude, latitude, marker='o', color='red', s=5)

m = Basemap(llcrnrlon=min(longitude)-2, llcrnrlat=min(latitude)-2,
            urcrnrlon=max(longitude)+2, urcrnrlat=max(latitude)+2, )
            # resolution='l', projection='merc', lat_0=np.mean(latitude), lon_0=np.mean(longitude))

m.drawcoastlines()
m.drawcountries()

plt.xlim(min(longitude)-2, max(longitude)+2)
plt.ylim(min(latitude)-2, max(latitude)+2)
plt.title('購入者の住所', fontproperties=fprop, fontsize=20 )
plt.savefig('geological_map.png' ,  dpi=200 )
plt.show()
plt.close()  
'''

#----------- deliver time vs. review ----------------------

'''
figure(num=None, figsize=(7, 5), dpi=100, facecolor='w', edgecolor='k') 
plt.scatter( df_payment['distance'], df_payment['expected_deliver_diff'],
            c=df_payment['review_score'], cmap='viridis', s=1 )

plt.xlabel('距離', fontproperties=fprop,)
plt.ylabel('配達予定日数との差', fontproperties=fprop,)
plt.xlim([ -2, 30])
plt.ylim([ -100, 100])
plt.title('配達距離の散布図', fontproperties=fprop, fontsize=20 )
plt.xticks(rotation=20)
cbar = plt.colorbar()
cbar.set_label('レビュースコア', fontproperties=fprop,)
plt.tight_layout()
plt.savefig('distance_scatter.png' ,  dpi=200 )
plt.show()
plt.close()
exit() 
'''

'''
figure(num=None, figsize=(7, 5), dpi=100, facecolor='w', edgecolor='k') 
plt.scatter( df_payment['payment_value'], df_payment['deliver_days'],
            c=df_payment['review_score'], cmap='viridis', s=1 )

plt.xlabel('商品価格', fontproperties=fprop,)
plt.ylabel('配達日数', fontproperties=fprop,)
plt.xlim([ -200, 4000])
plt.ylim([ -5, 150])
plt.title('　配達日数と商品価格の散布図', fontproperties=fprop, fontsize=20 )
plt.xticks(rotation=20)
cbar = plt.colorbar()
cbar.set_label('レビュースコア', fontproperties=fprop,)
plt.tight_layout()
plt.savefig('delivertime_review_scatter.png' ,  dpi=200 )
plt.show()
plt.close()
exit() 
'''

'''
figure(num=None, figsize=(7, 5), dpi=100, facecolor='w', edgecolor='k') 
plt.scatter( df_payment['expected_deliver_diff'], df_payment['deliver_days'],
            c=df_payment['review_score'], cmap='viridis', s=1 )

plt.xlabel('配達予定日との日数差', fontproperties=fprop,)
plt.ylabel('配達日数', fontproperties=fprop,)
plt.xlim([ -150, 150 ])
plt.ylim([ -5, 140])
plt.title('　配達予定日の日数差の散布図', fontproperties=fprop, fontsize=20 )
plt.xticks(rotation=20)
cbar = plt.colorbar()
cbar.set_label('レビュースコア', fontproperties=fprop,)
plt.tight_layout()
plt.savefig('expected_delivertime_review_scatter.png' ,  dpi=200 )
plt.show()
plt.close() 
'''

'''
figure(num=None, figsize=(7, 5), dpi=100, facecolor='w', edgecolor='k') 
plt.scatter( df_payment['order_shipped_days'], df_payment['expected_deliver_diff'],
            c=df_payment['review_score'], cmap='viridis', s=1 )

plt.xlabel('発送までの日数', fontproperties=fprop,)
plt.ylabel('配達予定の日数差', fontproperties=fprop,)
plt.xlim([ -5, 50 ])
plt.ylim([ -150, 100])
plt.title('配達予定日の日数差の散布図', fontproperties=fprop, fontsize=20 )
plt.xticks(rotation=20)
cbar = plt.colorbar()
cbar.set_label('レビュースコア', fontproperties=fprop,)
plt.tight_layout()
plt.savefig('shipping_delivertime_review_scatter.png' ,  dpi=200 )
plt.show()
plt.close() 
'''

'''
figure(num=None, figsize=(7, 5), dpi=100, facecolor='w', edgecolor='k') 
plt.scatter( df_payment['shipping_days'], df_payment['order_shipped_days'],
            c=df_payment['review_score'], cmap='viridis', s=1 )

plt.xlabel('発送日数', fontproperties=fprop,)
plt.ylabel('発送までの日数', fontproperties=fprop,)
plt.xlim([ -5, 150 ])
plt.ylim([ -5, 70])
plt.title('発送日数の散布図', fontproperties=fprop, fontsize=20 )
plt.xticks(rotation=20)
cbar = plt.colorbar()
cbar.set_label('レビュースコア', fontproperties=fprop,)
plt.tight_layout()
plt.savefig('shipping_delivertime_review2_scatter.png' ,  dpi=200 )
plt.show()
plt.close() 
'''

#------- review scores for product category ------------

grouped = df_payment.groupby('product_category_name_english')['review_score'].agg(['mean', 'std'])
worst_10 = grouped.sort_values('mean', ascending=False).tail(10)

figure(num=None, figsize=(7, 4), dpi=100, facecolor='w', edgecolor='k') 
#worst_10['mean'].plot(kind='bar', yerr=worst_10['std'], legend=False)
plt.barh(worst_10.index, worst_10['mean'], xerr=worst_10['std'], alpha=0.7, capsize=10, color=get_color_gradient("#8A5AC2", "#3575D5", len(worst_10)))
#plt.ylabel('Product Category')
plt.xlabel('レビュースコア', fontproperties=fprop,)
plt.title('平均レビュースコアのワースト１０カテゴリ', fontproperties=fprop, fontsize=16)
plt.tight_layout()
plt.savefig('worst10_review.png' ,  dpi=200 )
#plt.show()
plt.close()

#-------- seller delivery info -------------------------

'''
#seller = df_payment.groupby(['seller_id'])['expected_deliver_diff','order_shipped_days'].mean()
#seller['count'] =  df_payment.groupby(['seller_id'])['expected_deliver_diff'].count()

x = df_payment['order_shipped_days'].values
y = df_payment['expected_deliver_diff'].values
x = np.nan_to_num(x)
y = np.nan_to_num(y)
plt.scatter( x, y, s=10, c='gray'  )

x_grid = np.linspace(min(x), max(x), 100)
y_grid = np.linspace(min(y), max(y), 100)
X, Y = np.meshgrid(x_grid, y_grid)

Z = np.zeros_like(X)
for i in range(len(x)):
    Z += np.exp(-((X - x[i])**2 + (Y - y[i])**2))
    
#levels = np.linspace(0, np.max(Z), 10)
levels = [2, 15, 200] #np.linspace(5, 200, 5)
contour = plt.contour(X, Y, Z, '--', levels=levels, linewidths=1)

plt.colorbar(contour, label='Density',)

plt.xlabel('発送までの日数', fontproperties=fprop,)
plt.ylabel('配達予定の日数差', fontproperties=fprop,)
plt.xlim([ -2, 80 ])
#plt.ylim([ -5, 70])
plt.title('発送日数の散布図', fontproperties=fprop, fontsize=20 )
plt.xticks(rotation=20)
#cbar = plt.colorbar()
#cbar.set_label('販売回数', fontproperties=fprop,)
plt.tight_layout()
plt.savefig('seller_delivery_info_scatter.png' ,  dpi=200 )
plt.show()
plt.close() 
'''

#pdb.set_trace()

#---------- model ----------------------------------------------------------
#from transformers import BertTokenizer, BertForSequenceClassification
#import torch
#import tensorflow as tf
#import tensorflow_hub as hub
import textblob
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = df_payment.drop( columns=['order_id', 'payment_type', 'payment_installments', 
                            'payment_value', 'freight_value', 'review_score', 'seller_city',
                            'customer_id', 'order_status', 'order_purchase_timestamp',
                            'order_delivered_carrier_date', 'order_delivered_customer_date',
                            'order_estimated_delivery_date', 'customer_city', 'review_score_str',
                            'deliver_days', 'expected_deliver_diff', "shipping_days", 
                            ], axis=1 ).copy()
df.reset_index(drop=True, inplace=True)
gc.collect()

columns_to_standardize = ['price', 'order_shipped_days', 'distance']
scaler = StandardScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

# List of positive words.
positive_words = [ "recomendo", "excelete", "excelente", "maravilloso", "genial", "recomendado", "satisfecho", 
                    "contento", "bom", "otimo", "genial", "perfeito", "satisfeito", "satisfeita", 
                    "boa", "genial", "encantado", "feliz", "buen", "maravilloso", "Oltima", "positiva",    ]

def get_polarity(x):
    polarity1 = analyzer.polarity_scores(x['review_comment_title'])["compound"]
    polarity2 = analyzer.polarity_scores(x['review_comment_message'])["compound"]
    if polarity1>0 or polarity2>0:
        return 1
    else:
        for positive_word in positive_words:
            if re.search(positive_word, x['review_comment_title'].lower()):
                return 1
            elif re.search(positive_word, x['review_comment_message'].lower()):
                return 1
        return 0

analyzer = SentimentIntensityAnalyzer()
df['satisfied'] = df[['review_comment_title','review_comment_message']].fillna("").apply(lambda x: get_polarity(x), axis=1)


'''
#------ pytorch ------------------------------
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model_nlp = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels: 0 for not satisfied, 1 for satisfied

# Tokenize and encode the review comments
encoded_inputs = tokenizer(["satisfeito", "maravilhoso"], padding=True, truncation=True, return_tensors='pt')
# Perform sentiment analysis using the BERT model
with torch.no_grad():
    outputs = model_nlp(**encoded_inputs)
print("result = ", torch.argmax(outputs.logits, dim=1).tolist() )
exit()

# Function to label satisfaction based on review comment
def label_satisfaction(comment):
    if pd.isnull(comment):
        return -1  # Assign a default label for NaN values and empty comments
    else:
        inputs = tokenizer.encode_plus(comment, add_special_tokens=True, return_tensors='pt')
        outputs = model_nlp(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        return predicted_label
import time
t1 = time.time()
# Add a new column for satisfaction labels in the DataFrame
df['satisfaction_label'] = df['review_comment_message'].apply(label_satisfaction)
df['satisfaction_label2'] = df['review_comment_title'].apply(label_satisfaction)
print( time.time()-t1 )
pdb.set_trace()
#------------------------------------------------------------
'''

'''
# Load the USE module
module_url = "C:/Users/knkas/Desktop/ey_case_study/downloaded_TF_NLP_model/"
model_nlp = hub.load(module_url)

# Function to label satisfaction based on review comment
def label_satisfaction(comment):
    if pd.isnull(comment):
        return -1  # Assign a default label for NaN values
    else:
        embeddings = model_nlp([comment])
        prediction = tf.sigmoid(tf.keras.layers.Dense(1)(embeddings)).numpy()[0][0]
        return int(prediction >= 0.5)  # Using 0.5 as a threshold for satisfaction
import time
t1 = time.time()
# Add a new column for satisfaction labels in the DataFrame
df['satisfaction_label'] = df['review_comment_message'].apply(label_satisfaction)
df['satisfaction_label2'] = df['review_comment_title'].apply(label_satisfaction)
print( time.time()-t1 )
pdb.set_trace()
'''


df.drop( columns=['review_comment_title','review_comment_message'], axis=1, inplace=True )
X = df.drop(['satisfied'], axis=1)
y = df['satisfied']

categorical_cols = ['product_id', 'seller_id',  'customer_unique_id', 'product_category_name_english', 'customer_state', 'seller_state', ]
for col in categorical_cols:
    X[col] = X[col].astype('category')

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=23)


'''
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_cols)

params = {'objective': 'binary', 'metric': 'binary_logloss'}
model = lgb.train( params, lgb_train, num_boost_round=100 )
y_pred = model.predict(X_test)
y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
#print("Accuracy:", accuracy_score(y_test, y_pred))
'''

'''
customer_info = df_customer[['customer_unique_id','customer_state']]
seller_info = X[['seller_id','order_shipped_days']].groupby(['seller_id'], as_index=False).mean()
product_info = X[['product_id', 'price', 'product_category_name_english']]
test_data = pd.DataFrame()
for i in range(3):  #range(len(customer_info)):
    print(i)
    customer_id = customer_info.loc[i].customer_unique_id
    for k in range(len(product_info)):
        product_id = product_info.loc[k].product_id
        if product_id not in df.loc[ df.customer_unique_id==customer_id, "product_id" ].values:
            for l in range(len(seller_info)):
                data = pd.concat([customer_info.loc[i], product_info.loc[k], seller_info.loc[l]], axis=0).to_frame().transpose()                
                test_data = pd.concat([test_data, data], ignore_index=True) 
'''



#'''

def feval(preds, train_data):
    labels = train_data.get_label()
    preds_rounded = [1 if x >= 0.5 else 0 for x in preds]
    f1 = f1_score(labels, preds_rounded)
    return 'f1', f1, True  # metric_name, metric_value, is_higher_better
    
def objective(trial):
    params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        }

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_val],
                        verbose_eval=False,
                        num_boost_round=1000,  # Set a large number of maximum iterations
                        early_stopping_rounds=10,  # Specify the number of rounds without improvement for early stopping
                        feval=feval, # evaluation metric
                    )
                        
    y_pred = model.predict(X_val)
    y_pred = [1 if x >= 0.5 else 0 for x in y_pred]

    val = f1_score(y_val, y_pred)
    #val = precision_score(y_val, y_pred)
    #val = recall_score(y_val, y_pred)
    return val

# Step 5: Optimize Hyperparameters with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# Step 6: Retrain the Model with Best Hyperparameters
best_params = trial.params
lgb_train_val = lgb.Dataset(X_train_val, y_train_val)
model = lgb.train(best_params, lgb_train_val)

y_pred = model.predict(X_test)
y_pred = [1 if x >= 0.5 else 0 for x in y_pred]

f_score = f1_score(y_test, y_pred)
print("F-score on test set:", f_score)
#'''

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y_test)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(6, 4))
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png' ,  dpi=200 )
plt.show()
plt.close()

import pdb; pdb.set_trace()


qry = " select  *  from df_payment "
sqldf( qry, globals() )

