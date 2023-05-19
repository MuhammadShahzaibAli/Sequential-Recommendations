# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 01:15:45 2022

@author: Shahzaib
"""

import sys
import os 
import pandas as pd
import numpy as np

# Allows you to import the file from the previous exercises as a package
sys.path.append(os.path.abspath('C:\\Users\\Muhammad Shahzaib\\OneDrive - TUNI.fi\Desktop\\Student Docs\\Tampere University\\Period 2\\Recommender System'))

from lib import Collaborative_Filtering as cf
from lib import Group_Recommendations as gr

## Functions

### This fucntion calculates the unwatched movie ratings for the users and also provide a dataframe of normalized ratings
def norm_predictions(frame,users,sim_users):
    norm_pred=[]
    predictions=[]
    overall_mean = frame['rating'].mean()
    for i in users:
        pred_user=cf.pred_all_user(frame, i, sim_users)
        mean=frame[frame['userId']==i]['rating'].mean()
        std=np.std(frame[frame['userId']==i]['rating'])
        pred_user['normalized_rating']=(pred_user['rating']-mean)/std+overall_mean # Scoes ~ N(mu, 1)
        norm_pred.append(pred_user[['user','movie','normalized_rating']])
        predictions.append(pred_user[['user','movie','rating']])
    prediction_df=pd.concat(predictions)
    normalized_df=pd.concat(norm_pred)
    
    return prediction_df,normalized_df

def sequential(users,avg_df,misery_df,user_pred_norm,n):
    sequence=[avg_df.head(20)]
    group_diss = gr.group_dissatisfaction(users, user_pred_norm, avg_df.head(20))
    diss=[group_diss]
    iter_diss=[group_diss]
    cum_diss=group_diss
    
    avg_df=avg_df[~ avg_df['movie'].isin(avg_df.head(20)['movie'].values)] # remove previous recommendations before calculating new recomms
    misery_df=misery_df[~ misery_df['movie'].isin(avg_df.head(20)['movie'].values)]
    
    for i in range(2,n+1):
        data_df=gr.AAG(avg_df, misery_df, group_diss)
        group_diss=gr.group_dissatisfaction(users, user_pred_norm, data_df)
        cum_diss = cum_diss+group_diss
        avg_diss=cum_diss/i
        sequence.append(data_df)
        diss.append(group_diss)
        iter_diss.append(avg_diss)
        avg_df=avg_df[~ avg_df['movie'].isin(data_df['movie'].values)] # remove previous recommendations before calculating new recomms
        misery_df=misery_df[~ misery_df['movie'].isin(data_df['movie'].values)]
    return sequence,diss, iter_diss
        

# Main

## Reading Data
 
df=pd.read_csv('ratings.csv')
df=pd.DataFrame(df)

## Provide a list of users

arr = {3,4,18}

## Calculating similarity for all users

similarity_df = cf.sim_all_users(df)

## Calculating predictions & normalized predictions for all the unwatched movies by the users

user_predictions,user_predictions_normalized=norm_predictions(df,arr,similarity_df)

## Calculating the recommended movies through average aggregation

normalized_average_df=gr.average_method(user_predictions_normalized, arr)

## Calculating the recommended movies through least misery aggregation

normalized_least_misery_df=gr.least_misery(user_predictions_normalized, arr)

## Calculating group dissatisfaction

### Changing the column name of 'normalized_rating' to 'rating' to confrom it to the dissatisfaction function
user_predictions_normalized.columns=['user','movie','rating']

## Group recommendation using the new aggregation method and normalized average aggregated dataframe

# data_df=gr.AAG(normalized_average_df, normalized_least_misery_df, group_dissatisfaction_norm)


## Sequential group recommendation using normalized average aggregated and normalized least misery aggregated dataframe
recommendations,dissatisfactions, iter_diss =sequential(arr,normalized_average_df.copy(),normalized_least_misery_df.copy(),user_predictions_normalized.copy(),5)
print(recommendations)


