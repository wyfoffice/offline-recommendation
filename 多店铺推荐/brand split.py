#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:12:03 2018

@author: wyf
"""


import pandas as pd
import random
import numpy as np
import pickle

#retail_data = pd.read_csv('/home/wyf/Desktop/tmall/data.csv')
retail_data = pd.read_csv('raw_data.csv')
cleaned_retail = retail_data.dropna()
fortest = cleaned_retail.copy()

products = list(cleaned_retail.item_id.unique())
customers = list(np.sort(cleaned_retail.user_id.unique()))
cat = list(cleaned_retail.cat_id.unique())


def unpopular_goods(data, num):
    '''找出少于等于<=某一num的商品(冷门商品)'''
    data['Purchases'] = data.groupby(['item_id']).cumcount(ascending=False)+1
    data = data.drop_duplicates(subset=['item_id'], keep='first')
    goods = list(data[data['Purchases'] <= num].item_id)
    return goods
unpopular = unpopular_goods(cleaned_retail.copy(),7)

cleaned_retail = cleaned_retail[~cleaned_retail.item_id.isin(unpopular)]

cleaned_retail['Occurrence'] = cleaned_retail.groupby(['user_id','item_id']).cumcount(ascending=False)+1
df = cleaned_retail.drop_duplicates(subset=['user_id','item_id'], keep='first')

deleted_row = fortest.shape[0] - cleaned_retail.shape[0]
deleted_user = len(customers) - len(list(np.sort(cleaned_retail.user_id.unique())))
#df.to_csv('/home/wyf/Desktop/tmall/data.csv')

def many_brands(data, num):
    data = data.drop_duplicates(subset=['user_id','brand_id'],keep='first')
    data['user_brand'] = data.groupby(['user_id']).cumcount(ascending=False) + 1
    multi_brands = list(data[data['user_brand'] == num].user_id)
    return multi_brands
id_list = many_brands(df.copy(),3)

def bias_sample(id_list, pct_test):
    '''
    这里是对着上面many_brand找出来的跨店铺的购买者的划分
    我们把其中一个brand中的购买记录都归到test中，training set里抹除这些记录
    '''
    satisfied_user = id_list
    num_samples = int(np.ceil(pct_test*len(satisfied_user)))
    random.seed(1)
    samples = random.sample(satisfied_user, num_samples) # 这里返回的是user_id

    return samples

sampled_user_id = bias_sample(id_list, 0.2)

"""
def bias_sample(data, num, pct_test):
    '''
    num 是购买的item类别数，越大表示买的item越多（不同item）
    这个 bias_sample 是在购买数less than or 等于num的user中进行sample
    '''
    
#    df['times'] = df.groupby(['A']).cumcount(ascending=False)+1
#    df = df.drop_duplicates(subset=['A'], keep='first')
#    satisfied_user = set(df.A[df.times<=1])
#    # num_samples = int(np.ceil(pct_test*len(satisfied_user)))
#    # 
#    satisfied_user

    data['times'] = data.groupby(['user_id']).cumcount(ascending=False)+1
    data = data.drop_duplicates(subset=['user_id'], keep='first')
    satisfied_user = set(data.user_id[data.times<=num])
    num_samples = int(np.ceil(pct_test*len(satisfied_user)))
    random.seed(1)
    samples = random.sample(satisfied_user, num_samples) # 这里返回的是user_id

    return samples
"""
#sampled_user_id = bias_sample(df.copy(), 1, 0.2)

#def make_split(data, products):   # 经过测试, 没有问题
def make_split(data, sampled_user):   # 经过测试, 没有问题  
    
    training_set = data.copy()
    
#    test = data[data['user_id'].isin(sampled_user)]
#    test['Count'] = test.groupby(['A','B']).cumcount(ascending=False)+1
#    test = test.drop_duplicates(subset=['A','B'], keep='first')
#    test['rank'] = test['Count'].rank(ascending=False)
    respect_item = []
    for i in sampled_user:
        choosen_brand = training_set[training_set['user_id']==i][['item_id','brand_id']].iloc[0].brand_id
        user_item_brand = training_set[(training_set['brand_id']==choosen_brand) & (training_set['user_id']==i)][['user_id','item_id','brand_id']].values.tolist()
        for j in user_item_brand:
            '''
            在这里有些user，brand对会对应着多个item，这里我把它都分开，然后加进到test（respec_item里面）
            '''
            respect_item.append(j)
        training_set = training_set.drop(training_set[(training_set['brand_id']==choosen_brand) & (training_set['user_id']==i)].index)
#        print('the user is:\n', i)
#        print('item:\n',data['item_id'][data['user_id']==i].iloc[0])
#        print('before:\n',training_set[training_set.user_id==i])
        
        
#         print('before:\n',training_set[training_set.user_id==i])
#         print(training_set.iloc[[training_set[training_set.user_id==i].index[0]]])
        
#        print('after:\n',training_set[training_set.user_id==i])
    
#     user_item = list(zip(bias_samples_ind, sampled_customer_id, respect_item))

    return training_set, respect_item

product_train, product_users_altered = make_split(df.copy(), sampled_user_id)

#product_train.to_csv('/home/wyf/Desktop/tmall/train.csv')
product_train.to_csv('/home/wyf/Desktop/mixed/train.csv')
#with open('/home/wyf/Desktop/tmall/test.csv', 'w') as myfile:
#    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#    wr.writerow(product_users_altered)

#with open('/home/wyf/Desktop/tmall/test.txt', 'wb') as fp:
with open('/home/wyf/Desktop/mixed/test.txt', 'wb') as fp:
    pickle.dump(product_users_altered, fp)