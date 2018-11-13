# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:00:23 2018

@author: shagi
"""


import pandas as pd
import numpy as np
import time
import datetime

def load_data(file_name = 'train.csv'):
    path = '../../data/{}'.format(file_name)
    df = pd.read_csv(path)
    return df

def process_df(df,type_ ='train'):
    list_ids =list( df.Id)
    preds = list(df['Prediction'])
    ts = None
    if(type_ == 'train'):
        ts = int(time.mktime(datetime.datetime.strptime("2004-01-01","%Y-%m-%d").timetuple()))
    else:
        ts = int(time.mktime(datetime.datetime.strptime("2005-12-20","%Y-%m-%d").timetuple()))
    splitted = [a.split('_') for a in list_ids]
    users = [split[0][1:] for split in splitted]
    movies = [split[1][1:] for split in splitted]
    all_data = {}
    for i in range(len(preds)):
        if users[i] not in all_data.keys():
            all_data[users[i]] = []
        
        all_data[users[i]].append([movies[i],preds[i],ts]);
    
    return all_data


def combine_train_test():
    all_train = process_df(load_data());
    all_test = process_df(load_data('test.csv'),type_='test')
    
    for user, test_r in all_test.items():
        all_train[user].extend(test_r)    
    
    return all_train


def fix_preds():
    df = pd.read_table('./preds.txt',delim_whitespace  = True,header = None,names = ['user','movie','Prediction','dummy'] )
    df_test = load_data('test.csv')
    df['Id'] = list(map(lambda x,y: 'r'+str(x)+'_c'+str(y),df.user,df.movie))
    df.index=df.Id
    df=df.loc[df_test.Id]
    df['Prediction'] = df['Prediction'].round(0).astype(int)
    df.loc[:,'Prediction'][df['Prediction']>5] = 5
    df.loc[:,'Prediction'][df['Prediction']<=0] = 1
    df[['Id','Prediction']].to_csv('./preds_processed.csv',index = False)
    return df


if __name__=='__main__':
    fix_preds();    