#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:12:03 2018

@author: wyf
"""


import pandas as pd
import random
import numpy as np
import csv
import pickle

'''这个是用来把data划分成training set和test set'''

retail_data = pd.read_csv('/home/wyf/Desktop/tmall/data.csv')
#retail_data = pd.read_csv('/home/wyf/Desktop/tmall/8235/8235.csv')
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

def bias_sample(data, num, pct_test):
    '''
    num 是购买的item类别数，越大表示买的item越多（不同item）
    这个 bias_sample 是在购买数大于等于num的user中进行sample
    '''
    _data = data.copy()
    _data['times'] = _data.groupby(['user_id']).cumcount(ascending=False)+1
    satisfied_user = set(_data.user_id[_data.times>=num])
    num_samples = int(np.ceil(pct_test*len(satisfied_user)))
    random.seed(1)
    samples = random.sample(satisfied_user, num_samples) # 这里返回的是user_id

    return samples

sampled_user_id = bias_sample(df.copy(), 4, 0.2)

def make_split(data, sampled_user):   
    
    training_set = data.copy()

    respect_item = [] # 我们要被遮盖的item_id
    for i in sampled_user:
        respect_item.append(data['item_id'][data['user_id']==i].iloc[0])
        
        training_set = training_set.drop(training_set[training_set.user_id==i].index[0])

    
    user_item = list(zip(sampled_user, respect_item))


    return training_set, user_item

product_train, product_users_altered = make_split(df.copy(), sampled_user_id)


product_train.to_csv('/home/wyf/Desktop/tmall/8235/train.csv')


#with open('/home/wyf/Desktop/tmall/test.txt', 'wb') as fp:
with open('/home/wyf/Desktop/tmall/8235/test.txt', 'wb') as fp:
    pickle.dump(product_users_altered, fp)