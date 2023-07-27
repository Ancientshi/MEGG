import os
import sys
import random
import pandas as pd
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from config import *

seed=args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
g = torch.Generator()
g.manual_seed(seed)

root_path='/home/yunxshi/Data/workspace/MEGG'


def get_merged_data(dataset='ml-20m'):
    if dataset == 'ml-1m':
        dataset_path='/home/yunxshi/Data/ml-1m'
        rating_file = os.path.join(dataset_path, 'ratings.dat')
        user_file = os.path.join(dataset_path, 'users.dat')
        movie_file = os.path.join(dataset_path, 'movies.dat')
        
        # Load data
        ratings = pd.read_csv(rating_file, sep='::', engine='python', encoding='latin-1', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        
        if args.task=='binary':
            ratings['rating'] = ratings['rating'].apply(lambda x: 1 if x>=4 else 0)
            
        users = pd.read_csv(user_file, sep='::', engine='python', encoding='latin-1', names=['user_id', 'gender', 'age', 'occupation', 'zip'])
        movies = pd.read_csv(movie_file, sep='::', engine='python', encoding='latin-1', names=['movie_id', 'title', 'genres'])

        # Merge data
        data = pd.merge(pd.merge(ratings, users), movies)
        data = data.sort_values(by='timestamp')
        data.reset_index(drop=True, inplace=True)
        columns=data.columns.tolist()
        return data, columns

    elif dataset=='ml-20m':
        dataset_path='/home/yunxshi/Data/ml-20m'
        rating_file = os.path.join(dataset_path, 'ratings.csv')
        movie_file = os.path.join(dataset_path, 'movies.csv')
        
        # Load data
        ratings = pd.read_csv(rating_file)
        movies = pd.read_csv(movie_file)

        ratings['rating'] = ratings['rating'].astype(float)
        data = pd.merge(ratings, movies)
        data = data.sort_values(by='timestamp')
        data.reset_index(drop=True, inplace=True)
        columns=data.columns.tolist()
        return data, columns
    
    elif dataset=='taobao2014':
        dataset_path='/home/yunxshi/Data/taobao2014'
        interaction_file = os.path.join(dataset_path, 'tianchi_mobile_recommend_train_user.csv')
        
        # Load data
        data = pd.read_csv(interaction_file)

        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H')
        

        begin=5000
        active_users = data['user_id'].value_counts().index.tolist()
        active_users=active_users[begin:begin+1000]
        data = data[data['user_id'].isin(active_users)]

        data = data.sort_values(by='time')
        data.reset_index(drop=True, inplace=True)
        columns=data.columns.tolist()
        return data, columns
    
    elif dataset=='lastfm-1k':
        dataset_path='/home/yunxshi/Data/lastfm-dataset-1K'
        user_file = os.path.join(dataset_path, 'userid-profile.tsv')
        interaction_file=os.path.join(dataset_path, 'userid-timestamp-artid-artname-traid-traname.tsv')
        
        users = pd.read_csv(user_file,sep='\t',skiprows=1,header=None,names=['userid','gender','age','country','signup'])
        users["signup"] = pd.to_datetime(users["signup"], format="%b %d, %Y")
        users["signup_month"] = users["signup"].dt.month
        users["signup_year"] = users["signup"].dt.year
        users['signup_day'] = users['signup'].dt.day
        users.drop(['signup'],axis=1,inplace=True)

        users['gender'].fillna(0, inplace=True)
        users['gender'] = users['gender'].replace({'m': 1, 'f': -1})

        for f in ['age', 'signup_month','signup_year','signup_day']:
            users[f].fillna(-1, inplace=True)
            users[f] = users[f].replace('', -1)
        users['country'].fillna('unknown', inplace=True)
        users['country'] = users['country'].replace('', 'unknown')

        interactions = pd.read_csv(interaction_file,sep='\t',skiprows=1,header=None,names=['userid','timestamp','artid','artname','traid','traname'],  error_bad_lines=False)
        interactions.drop(['artname','traname'],axis=1,inplace=True)
        interactions.dropna(inplace=True)
        
        interactions['timestamp'] =pd.to_datetime(interactions['timestamp'])
        interactions['month'] = interactions['timestamp'].dt.month
        interactions['year'] = interactions['timestamp'].dt.year
        interactions['day'] = interactions['timestamp'].dt.day

        interactions['like'] = interactions.groupby(['userid', 'traid'])['userid'].transform('count')
        
        # 1: interactions['like'] 为0
        # 2: interactions['like'] 为1-4
        # 3: interactions['like'] 为5-14
        # 4: interactions['like'] 为15-29
        # 5: interactions['like'] 为30-..
        conditions = [interactions['like'] >= 30, interactions['like'].between(15, 29), interactions['like'].between(5, 14), interactions['like'].between(1, 4), interactions['like'] == 0 ]
        choices = [5,4,3, 2, 1]
        interactions['like'] = np.select(conditions, choices, default=0)

        min_timestamp=interactions['timestamp'].min()
        max_timestamp=min_timestamp+pd.DateOffset(months=12)
        interactions = interactions[(interactions['timestamp']>=min_timestamp) & (interactions['timestamp']<=max_timestamp)]
        
        
        data = pd.merge(interactions, users, on='userid')
        data = data.sort_values(by='timestamp')
        data.reset_index(drop=True, inplace=True)
        columns=data.columns.tolist()
        return data, columns
    
    elif dataset=='lastfm-1k_CTR':
        dataset_path='/home/yunxshi/Data/lastfm-dataset-1K'
        user_file = os.path.join(dataset_path, 'userid-profile.tsv')
        interaction_file=os.path.join(dataset_path, 'userid-timestamp-artid-artname-traid-traname.tsv')
        
        users = pd.read_csv(user_file,sep='\t',skiprows=1,header=None,names=['userid','gender','age','country','signup'])
        users["signup"] = pd.to_datetime(users["signup"], format="%b %d, %Y")
        users["signup_month"] = users["signup"].dt.month
        users["signup_year"] = users["signup"].dt.year
        users['signup_day'] = users['signup'].dt.day
        users.drop(['signup'],axis=1,inplace=True)

        users['gender'].fillna(0, inplace=True)
        users['gender'] = users['gender'].replace({'m': 1, 'f': -1})

        for f in ['age', 'signup_month','signup_year','signup_day']:
            users[f].fillna(-1, inplace=True)
            users[f] = users[f].replace('', -1)
        users['country'].fillna('unknown', inplace=True)
        users['country'] = users['country'].replace('', 'unknown')

        
        
        interactions = pd.read_csv(interaction_file,sep='\t',skiprows=1,header=None,names=['userid','timestamp','artid','artname','traid','traname'])
        interactions.drop(['artname','traname'],axis=1,inplace=True)
        interactions.dropna(inplace=True)
        
        interactions['timestamp'] =pd.to_datetime(interactions['timestamp'])
        interactions['month'] = interactions['timestamp'].dt.month
        interactions['year'] = interactions['timestamp'].dt.year
        interactions['day'] = interactions['timestamp'].dt.day

        interactions['like'] = interactions.groupby(['userid', 'traid'])['userid'].transform('count')


        if args.sample_size=='12months':
            min_timestamp=interactions['timestamp'].min()
            max_timestamp=min_timestamp+pd.DateOffset(months=12)
            interactions = interactions[(interactions['timestamp']>=min_timestamp) & (interactions['timestamp']<=max_timestamp)]
        elif args.sample_size=='36months':
            min_timestamp=interactions['timestamp'].min()
            max_timestamp=min_timestamp+pd.DateOffset(months=36)
            interactions = interactions[(interactions['timestamp']>=min_timestamp) & (interactions['timestamp']<=max_timestamp)]
            
        elif args.sample_size=='500w':
            interactions = interactions.sample(n=5000000)
        elif args.sample_size=='original':
            pass
        
        data = pd.merge(interactions, users, on='userid')
        data = data.sort_values(by='timestamp')
        data.reset_index(drop=True, inplace=True)
        columns=data.columns.tolist()
        return data, columns
    
    
    elif dataset=='douban':
        dataset_path='/home/yunxshi/Data/douban'
        itemfeat_file=os.path.join(dataset_path, 'item_feat.csv')
        itemfeat = pd.read_csv(itemfeat_file,sep=' ',header=None)

        itemfeat['feat']=itemfeat.apply(lambda x: '|'.join([str(i) for i in x.index if x[i]==1]),axis=1)
        itemfeat=itemfeat[['feat']]
        itemfeat['item_id']=itemfeat.index

        interaction_file = os.path.join(dataset_path, 'ratings.csv')
        interactions = pd.read_csv(interaction_file,header=None,sep='\t',names=['user_id','item_id','rating','timestamp','year'])
        
        if args.task=='binary':
            interactions['rating'] = interactions['rating'].apply(lambda x: 1 if x>=4 else 0)
        
        data = pd.merge(interactions, itemfeat, on='item_id')
        data = data.sort_values(by='timestamp')
        data.reset_index(drop=True, inplace=True)
        columns=data.columns.tolist()
        return data, columns





