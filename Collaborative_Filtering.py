# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 04:06:55 2022

@author: Shahzaib
"""

import pandas as pd
import numpy as np

# User Based Collaborative Filtering

## Functions For User Based Filtering

#### Function used to calculate Pearson Correlation between all users

def sim_all_users(frame):

    merged_df=pd.merge(frame,frame,how='inner',on='movieId')
    
    features_df=merged_df[['userId_x','rating_x','userId_y','rating_y']].groupby(['userId_x', 'userId_y']).mean().reset_index()
    features_df.columns=['userId_x', 'userId_y','rating_mean_x', 'rating_mean_y']
    
    data_df=pd.merge(merged_df,features_df,how='inner',on=('userId_x','userId_y'))
    data_df=data_df[['userId_x', 'userId_y','movieId','rating_x','rating_y','rating_mean_x', 'rating_mean_y']]
    
    data_df['(r_a-r_a_mean)']=data_df['rating_x']-data_df['rating_mean_x']
    data_df['(r_b-r_b_mean)']=data_df['rating_y']-data_df['rating_mean_y']
    data_df['num']=(data_df['(r_a-r_a_mean)'])*(data_df['(r_b-r_b_mean)'])
    
    data_df['(r_a-r_a_mean)^2']=(data_df['rating_x']-data_df['rating_mean_x'])*(data_df['rating_x']-data_df['rating_mean_x'])
    data_df['(r_b-r_b_mean)^2']=(data_df['rating_y']-data_df['rating_mean_y'])*(data_df['rating_y']-data_df['rating_mean_y'])
    
    dataframe_df=data_df[['userId_x','userId_y','num','(r_a-r_a_mean)^2','(r_b-r_b_mean)^2']].groupby(['userId_x','userId_y']).sum().reset_index()
    dataframe_df.columns=['userId_x', 'userId_y','num', 'sum((r_a-r_a_mean)^2)','sum((r_b-r_b_mean)^2)']
    dataframe_df['similarity']=((dataframe_df['num'])/(((dataframe_df['sum((r_a-r_a_mean)^2)'])**(1/2))*((dataframe_df['sum((r_b-r_b_mean)^2)'])**(1/2))))
    dataframe_df=dataframe_df.fillna(0)
    
    return dataframe_df[['userId_x','userId_y','similarity']]


### Function to calculate the similarities between 2 users

def sim_users(a,b,s_df):
    return (s_df[(s_df['userId_x']==a) & (s_df['userId_y']==b)]['similarity'].values[0])

### Funtion to provide the rating of a movie for a user

def pred_user(frame,a,movie,similarity_df):
    df_rating_mean=frame[frame['userId']==a]['rating'].mean()
    
    sim_user=similarity_df[similarity_df['userId_x']==a]
    
    df_movie=frame[frame['movieId']==movie]
    
    merged_df=pd.merge(df_movie,sim_user,how='inner',left_on='userId',right_on='userId_y')
    merged_df=merged_df[['userId_x','movieId','rating','userId','similarity']]
    sum_sim=sum(abs(merged_df['similarity']))
    
    users_means=frame[['userId','rating']].groupby(['userId']).mean().reset_index()
    users_means.columns=['userId','all_rating_mean']
    
    data_df=pd.merge(merged_df,users_means,how='inner',on='userId')
    
    data_df['rb-rb_mean']=(data_df['rating'])-(data_df['all_rating_mean'])
    data_df['num']=(data_df['similarity'])*(data_df['rb-rb_mean'])
    
    num=sum(data_df['num'])
    if sum_sim==0:
        p=0
    else:
        p=df_rating_mean+(num/sum_sim)
    return pd.DataFrame([(a,movie,p)],columns=['user','movie','rating'])

### Funtion to provide the rating of all unwatched movies by a single user

### Assumption: We only pass on unwatched movies by a user in the prediction function for a user

def pred_all_user(f_df,a,s_df):
    df_1=f_df[f_df['userId']==a]
    df_movie=pd.DataFrame(f_df['movieId'].unique(),columns=['movieId'])
    df_joined=pd.merge(df_1,df_movie,how='outer',on='movieId')
    df_unwatched=df_joined[df_joined['userId'].isna()]
    return pd.concat([pred_user(f_df,a,j,s_df) for j in set(df_unwatched['movieId'])])







# Item Based Collaborative Filtering 

## Funtions of Item Based Collaborative Filterning

### Function to Calculate Cosine Similarity between all Items

def sim_items_all(frame):

    merged_df=pd.merge(frame,frame,how='inner',on='userId')
    user_mean_df=frame[['userId','rating']].groupby(['userId']).mean().reset_index()
    user_mean_df.columns=['userId','user_rating_mean']
    
    data_df=pd.merge(merged_df,user_mean_df,how='inner',on='userId')
    data_df=data_df[['movieId_x', 'movieId_y','userId','rating_x','rating_y','user_rating_mean']]
    
    data_df['(r_a-r_mean)']=data_df['rating_x']-data_df['user_rating_mean']
    data_df['(r_b-r_mean)']=data_df['rating_y']-data_df['user_rating_mean']
    data_df['num']=(data_df['(r_a-r_mean)'])*(data_df['(r_b-r_mean)'])
    
    data_df['(r_a-r_mean)^2']=(data_df['rating_x']-data_df['user_rating_mean'])*(data_df['rating_x']-data_df['user_rating_mean'])
    data_df['(r_b-r_mean)^2']=(data_df['rating_y']-data_df['user_rating_mean'])*(data_df['rating_y']-data_df['user_rating_mean'])
    
    dataframe_df=data_df[['movieId_x','movieId_y','num','(r_a-r_mean)^2','(r_b-r_mean)^2']].groupby(['movieId_x','movieId_y']).sum().reset_index()
    dataframe_df.columns=['movieId_x', 'movieId_y','num', 'sum((r_a-r_mean)^2)','sum((r_b-r_mean)^2)']
    dataframe_df['similarity']=((dataframe_df['num'])/(((dataframe_df['sum((r_a-r_mean)^2)'])**(1/2))*((dataframe_df['sum((r_b-r_mean)^2)'])**(1/2))))
    dataframe_df=dataframe_df.fillna(0)
    
    return dataframe_df[['movieId_x','movieId_y','similarity']]

### Function to Calculate Cosine Similarity between Items

def sim_items(a,b,st_df):
    return st_df[(st_df['movieId_x']==a) & (st_df['movieId_y']==b)]['similarity'].values[0]


### Assumption: we only provide prediction function with movie id's that the user hasn't watched

### Prediction Function for to calculate the rating of a single user for a specific movie

def pred_item(frame,a,movie,similarity_df):
     
    df_user=frame[frame['userId']==a]
    df_user=df_user[['userId','movieId','rating']]
    df_req_mov=similarity_df[similarity_df['movieId_x']==movie]
    df_merge_pred=pd.merge(df_user,df_req_mov,how='inner',left_on=('movieId'),right_on=('movieId_y'))

    den=sum(abs(df_merge_pred['similarity']))

    df_merge_pred['sim*rb']=(df_merge_pred['similarity'])*(df_merge_pred['rating'])
    num=sum(df_merge_pred['sim*rb'])
    if den==0:
        return pd.DataFrame([(a,movie,0)],columns=['user','movie','rating'])
    else:
        return pd.DataFrame([(a,movie,abs(num/den))],columns=['user','movie','rating'])

### Function to calculate ratings for all the unwatched movies by a single user

def pred_all_item(f_df,a,s_df):
    df_1=f_df[f_df['userId']==a]
    df_user=pd.DataFrame(df['movieId'].unique(),columns=['movieId'])
    df_joined=pd.merge(df_1,df_user,how='outer',on='movieId')
    df_unwatched=df_joined[df_joined['userId'].isna()]
    return pd.concat([pred_item(f_df,a,j,s_df) for j in set(df_unwatched['movieId'])])

if __name__=='__main__':
    ## Reading Data
    
    df=pd.read_csv('ratings.csv')
    df=pd.DataFrame(df)
    print(df.head(10))
    print(df.count())
    
    
    ## Main
    
    ### User based Collaborative Filtering Checks
    
    ### Calculating the similarity between all users
    
    similarity_users=sim_all_users(df)
    
    ### Checking the similarity between 2 users
    
    print(sim_users(1, 1, similarity_users))
    
    ### Prediction of all the ratings for unwatched movies by a specific user
    
    prediction_userbased = pred_all_user(df, 1, similarity_users)
    
    # 10 most similar user's to user 1 based on userbased collaborative filtering
    
    most_similar_userbased=similarity_users[similarity_users['userId_x']==1].sort_values(by='similarity',ascending=False)
    print(most_similar_userbased.head(10))
    
    # 20 most relevent movies to user 1 based on userbased collaborative filtering
    
    most_relevent_userbased=prediction_userbased.sort_values(by='rating',ascending=False)
    print(most_relevent_userbased.head(20))
    
    
    
    ### Item Based Collaborative Filtering Checks
    
    ### Calculating the similarity between all Items
    
    similarity_Items=sim_items_all(df)
    
    ### Checking the similarity between 2 Items
    
    print(sim_items(2, 3, similarity_Items))
    
    ### Prediction of all the ratings for unwatched movies by a specific user
    
    prediction_itembased = pred_all_item(df, 2, similarity_Items)
    
    # 10 most similar movies to movie 2 based on itembased collaborative filtering
    
    most_similar_itembased=similarity_Items[similarity_Items['movieId_x']==2].sort_values(by='similarity',ascending=False)
    print(most_similar_itembased.head(10))
    
    # 20 most relevent movies to user 2 based on Itembased collaborative filtering
    
    most_relevent_itembased=prediction_itembased.sort_values(by='rating',ascending=False)
    print(most_relevent_itembased.head(20))
    


