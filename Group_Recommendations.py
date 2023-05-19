# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 03:15:14 2022

@author: Shahzaib
"""

import sys
import os 
import pandas as pd
import numpy as np

# Allows you to import the file from the first exercise as a package
sys.path.append(os.path.abspath('C:\\Users\\Muhammad Shahzaib\\OneDrive - TUNI.fi\Desktop\\Student Docs\\Tampere University\\Period 2\\Recommender System'))

from lib import Collaborative_Filtering as cf


## This fucntion gives dataframe of prediction ratings for all users

def users_pred_rating(frame,users,sim_users):
    predictions=[]
    for i,ival in enumerate(users):
        pred_user=cf.pred_all_user(frame, ival, sim_users)
        predictions.append(pred_user)
    prediction_df=pd.concat(predictions)
    return prediction_df
    

## Calculating Dissatisfaction

def group_dissatisfaction(users,user_pred,group_ratings):
    user_satisfaction=[]
    for i,ival in enumerate(users):
        den_df=user_pred[user_pred['user']==ival].sort_values(by='rating',ascending=False).head(20)
        den=den_df['rating'].sum()
        # num_df=den_df[den_df['movie'].isin(group_ratings['movie'])]
        num_df=user_pred[(user_pred['user']==ival) & (user_pred['movie'].isin(group_ratings.nlargest(20, columns='rating')['movie']))]
        num=num_df['rating'].sum()
        sat=num/den
        user_satisfaction.append(sat)
    diss=max(user_satisfaction)-min(user_satisfaction)
    return diss
        
        

# Average Method

## Function
def average_method(pred,users):
    # predictions=[]
    # for i,ival in enumerate(users):
    #     pred_user=cf.pred_all_user(frame, ival, sim_users)
    #     predictions.append(pred_user)
        
    # prediction_df=pd.concat(predictions)
    data=pred.groupby(['movie']).agg(['mean','count']).reset_index()
    data.columns=['movie','user_mean','user_count','rating','rating_count']
    data=data[data['rating_count']==len(users)]
    data=data[['movie','rating']]

    return data.sort_values(by='rating',ascending=False)

# Least Misery Method

def least_misery(pred,users):
    # predictions=[]
    # for i,ival in enumerate(users):
    #     pred_user=cf.pred_all_user(frame, ival, sim_users)
    #     predictions.append(pred_user)
        
    # prediction_df=pd.concat(predictions)
    data=pred.groupby(['movie']).agg(['min','count']).reset_index()
    data.columns=['movie','user_min','user_count','rating','rating_count']
    data=data[data['rating_count']==len(users)]
    data=data[['movie','rating']]

    return data.sort_values(by='rating',ascending=False)

# def group_rating(frame,users,sim_users,agg_func='mean'):
#     predictions=[]
#     for i,ival in enumerate(users):
#         pred_user=cf.pred_all_user(frame, ival, sim_users)
#         predictions.append(pred_user)
        
#     prediction_df=pd.concat(predictions)
#     data=prediction_df[['movie','rating']].groupby(['movie']).agg([agg_func]).reset_index()

#     return data.sort_values(by='rating',ascending=False).head(20)

## New Agreegation Method

def AAG(avg_df,least_df,alpha):
    score_df=pd.merge(avg_df,least_df,how='inner',on='movie')
    score_df.columns=['movie','avg_rating','least_rating']
    score_df['score']=((1-alpha)*(score_df['avg_rating']))+((alpha)*(score_df['least_rating']))
    score_df=score_df.nlargest(20, columns='score')
    score_df=score_df[['movie','score']]
    score_df.columns=['movie','rating']
    return score_df

if __name__=='__main__':
    
    ## Reading Data
    
    df=pd.read_csv('ratings.csv')
    df=pd.DataFrame(df)
    
    ## provide a list of users
    
    arr={1,2,3}
    
    similarity_df=cf.sim_all_users(df)
    users_predictions=users_pred_rating(df,arr,similarity_df)
    
    ## Top 20 movies by average method
    
    Avg_df=average_method(users_predictions,arr)
    Avg_df_top=Avg_df.head(20)
    
    ## Top 20 movies by least misery method
    
    least_misery_df=least_misery(users_predictions,arr)
    least_misery_df_top=least_misery_df.head(20)
    
    
    ## Calculating Dissatisfaction using users, user predictions and top 20 group recommendations from average method
    
    group_dis=group_dissatisfaction(arr,users_predictions,least_misery_df_top)
    
    ## New Agreegation Method
    
    aag_df=AAG(Avg_df,least_misery_df,group_dis)
    aag_df=aag_df[['movie','score']]



