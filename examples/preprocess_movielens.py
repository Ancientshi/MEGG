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
            #将rating为4,5的值转换为1，其他为0
            ratings['rating'] = ratings['rating'].apply(lambda x: 1 if x>=4 else 0)
            
        users = pd.read_csv(user_file, sep='::', engine='python', encoding='latin-1', names=['user_id', 'gender', 'age', 'occupation', 'zip'])
        movies = pd.read_csv(movie_file, sep='::', engine='python', encoding='latin-1', names=['movie_id', 'title', 'genres'])
        
        # genres_dict = {'Action':1, 'Adventure':2, 'Animation':3, 'Children\'s':4, 'Comedy':5, 'Crime':6, 'Documentary':7, 'Drama':8, 'Fantasy':9, 'Film-Noir':10, 'Horror':11, 'Musical':12, 'Mystery':13, 'Romance':14, 'Sci-Fi':15, 'Thriller':16, 'War':17, 'Western':18}
        # movies['genres'] = movies['genres'].apply(lambda x: [genres_dict[i] for i in x.split('|')])
        # movies['genres'] = movies['genres'].apply(lambda x: int(''.join([str(i) for i in x])))

        # Merge data
        data = pd.merge(pd.merge(ratings, users), movies)
        #按照时间戳排序
        data = data.sort_values(by='timestamp')
        #重置索引
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

        #rating列的数据类型转换为string
        ratings['rating'] = ratings['rating'].astype(float)
        
        # movies left join ratings
        data = pd.merge(ratings, movies)
        #按照时间戳排序
        data = data.sort_values(by='timestamp')
        #重置索引
        data.reset_index(drop=True, inplace=True)
        columns=data.columns.tolist()
        return data, columns
    
    elif dataset=='taobao2014':
        dataset_path='/home/yunxshi/Data/taobao2014'
        interaction_file = os.path.join(dataset_path, 'tianchi_mobile_recommend_train_user.csv')
        
        # Load data
        data = pd.read_csv(interaction_file)

        # #将time列的2014-12-06 02数据类型转换为datetime
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H')
        

        #找出最活跃的1000个用户 user_id
        begin=5000
        active_users = data['user_id'].value_counts().index.tolist()
        active_users=active_users[begin:begin+1000]
        #找出最活跃的1000个用户的数据
        data = data[data['user_id'].isin(active_users)]

        
        #按照时间戳排序
        data = data.sort_values(by='time')
        #重置索引
        data.reset_index(drop=True, inplace=True)
        columns=data.columns.tolist()
        return data, columns
    
    elif dataset=='lastfm-1k':
        dataset_path='/home/yunxshi/Data/lastfm-dataset-1K'
        user_file = os.path.join(dataset_path, 'userid-profile.tsv')
        interaction_file=os.path.join(dataset_path, 'userid-timestamp-artid-artname-traid-traname.tsv')
        
        #读取user_file,分隔符为\t,忽略掉第一行,userid \t gender ('m'|'f'|empty) \t age (int|empty) \t country (str|empty) \t signup (date|empty)
        users = pd.read_csv(user_file,sep='\t',skiprows=1,header=None,names=['userid','gender','age','country','signup'])
        #根据signup列构建column: signup_month,signup_year，signup列的数据格式为xxx xx, xxxx
        # 将signup列转换为datetime类型
        users["signup"] = pd.to_datetime(users["signup"], format="%b %d, %Y")
        # 从signup列中提取月份和年份数据，并创建新的列
        users["signup_month"] = users["signup"].dt.month
        users["signup_year"] = users["signup"].dt.year
        users['signup_day'] = users['signup'].dt.day
        users.drop(['signup'],axis=1,inplace=True)

        #将gender为空值的转为-1
        users['gender'].fillna(0, inplace=True)
        #将gender为m的值转换为1，f的值转换为-1
        users['gender'] = users['gender'].replace({'m': 1, 'f': -1})

        for f in ['age', 'signup_month','signup_year','signup_day']:
            #将age列的空值或者''转换为-1
            users[f].fillna(-1, inplace=True)
            users[f] = users[f].replace('', -1)
        users['country'].fillna('unknown', inplace=True)
        users['country'] = users['country'].replace('', 'unknown')

              
        #读取interaction_file,分隔符为\t,忽略掉第一行, 
        interactions = pd.read_csv(interaction_file,sep='\t',skiprows=1,header=None,names=['userid','timestamp','artid','artname','traid','traname'],  error_bad_lines=False)
        #删掉artname列与traname列
        interactions.drop(['artname','traname'],axis=1,inplace=True)
        #删掉interactions中有空值的行
        interactions.dropna(inplace=True)
        
        interactions['timestamp'] =pd.to_datetime(interactions['timestamp'])
        interactions['month'] = interactions['timestamp'].dt.month
        interactions['year'] = interactions['timestamp'].dt.year
        interactions['day'] = interactions['timestamp'].dt.day

        # 根据 userid 和 traid 进行分组，并计算每组的交互次数
        interactions['like'] = interactions.groupby(['userid', 'traid'])['userid'].transform('count')

        # 新建 like 列，根据交互次数进行赋值
        # 根据交互次数给like赋值
        # 1: interactions['like'] 为0
        # 2: interactions['like'] 为1-4
        # 3: interactions['like'] 为5-14
        # 4: interactions['like'] 为15-29
        # 5: interactions['like'] 为30-..
        conditions = [interactions['like'] >= 30, interactions['like'].between(15, 29), interactions['like'].between(5, 14), interactions['like'].between(1, 4), interactions['like'] == 0 ]
        choices = [5,4,3, 2, 1]
        interactions['like'] = np.select(conditions, choices, default=0)

        # #绘制出like的分布图，保存在like.png中
        # interactions['like'].value_counts().plot(kind='bar')
        # plt.savefig('/home/yunxshi/Data/lastfm-dataset-1K/like.png')

        # if args.sample_size=='12months':
        #     #从最小的timestamp开始，从中取出12个月的数据
        #     min_timestamp=interactions['timestamp'].min()
        #     #加上6个月的时间
        #     max_timestamp=min_timestamp+pd.DateOffset(months=12)
        #     #从interactions中取出时间在min_timestamp和max_timestamp之间的数据 
        #     interactions = interactions[(interactions['timestamp']>=min_timestamp) & (interactions['timestamp']<=max_timestamp)]
        # elif args.sample_size=='36months':
        #     #从最小的timestamp开始，从中取出36个月的数据
        #     min_timestamp=interactions['timestamp'].min()
        #     #加上6个月的时间
        #     max_timestamp=min_timestamp+pd.DateOffset(months=36)
        #     #从interactions中取出时间在min_timestamp和max_timestamp之间的数据 
        #     interactions = interactions[(interactions['timestamp']>=min_timestamp) & (interactions['timestamp']<=max_timestamp)]
        # elif args.sample_size=='500w':
        #     #从interactions中随机提取500w条数据
        #     interactions = interactions.sample(n=5000000)
        # elif args.sample_size=='original':
        #     pass
        
        #从最小的timestamp开始，从中取出12个月的数据
        min_timestamp=interactions['timestamp'].min()
        #加上6个月的时间
        max_timestamp=min_timestamp+pd.DateOffset(months=12)
        #从interactions中取出时间在min_timestamp和max_timestamp之间的数据 
        interactions = interactions[(interactions['timestamp']>=min_timestamp) & (interactions['timestamp']<=max_timestamp)]
        
        
        # Merge data,将interactions和itemfeat合并，合并的列是item_id
        data = pd.merge(interactions, users, on='userid')
        #按照时间戳排序
        data = data.sort_values(by='timestamp')
        #重置索引
        data.reset_index(drop=True, inplace=True)
        columns=data.columns.tolist()
        return data, columns
    
    elif dataset=='lastfm-1k_CTR':
        dataset_path='/home/yunxshi/Data/lastfm-dataset-1K'
        user_file = os.path.join(dataset_path, 'userid-profile.tsv')
        interaction_file=os.path.join(dataset_path, 'userid-timestamp-artid-artname-traid-traname.tsv')
        
        #读取user_file,分隔符为\t,忽略掉第一行,userid \t gender ('m'|'f'|empty) \t age (int|empty) \t country (str|empty) \t signup (date|empty)
        users = pd.read_csv(user_file,sep='\t',skiprows=1,header=None,names=['userid','gender','age','country','signup'])
        #根据signup列构建column: signup_month,signup_year，signup列的数据格式为xxx xx, xxxx
        # 将signup列转换为datetime类型
        users["signup"] = pd.to_datetime(users["signup"], format="%b %d, %Y")
        # 从signup列中提取月份和年份数据，并创建新的列
        users["signup_month"] = users["signup"].dt.month
        users["signup_year"] = users["signup"].dt.year
        users['signup_day'] = users['signup'].dt.day
        users.drop(['signup'],axis=1,inplace=True)

        #将gender为空值的转为-1
        users['gender'].fillna(0, inplace=True)
        #将gender为m的值转换为1，f的值转换为-1
        users['gender'] = users['gender'].replace({'m': 1, 'f': -1})

        for f in ['age', 'signup_month','signup_year','signup_day']:
            #将age列的空值或者''转换为-1
            users[f].fillna(-1, inplace=True)
            users[f] = users[f].replace('', -1)
        users['country'].fillna('unknown', inplace=True)
        users['country'] = users['country'].replace('', 'unknown')

        
        
        #读取interaction_file,分隔符为\t,忽略掉第一行, 
        interactions = pd.read_csv(interaction_file,sep='\t',skiprows=1,header=None,names=['userid','timestamp','artid','artname','traid','traname'])
        #删掉artname列与traname列
        interactions.drop(['artname','traname'],axis=1,inplace=True)
        #删掉interactions中有空值的行
        interactions.dropna(inplace=True)
        
        interactions['timestamp'] =pd.to_datetime(interactions['timestamp'])
        interactions['month'] = interactions['timestamp'].dt.month
        interactions['year'] = interactions['timestamp'].dt.year
        interactions['day'] = interactions['timestamp'].dt.day

        # 根据 userid 和 traid 进行分组，并计算每组的交互次数
        interactions['like'] = interactions.groupby(['userid', 'traid'])['userid'].transform('count')


        if args.sample_size=='12months':
            #从最小的timestamp开始，从中取出12个月的数据
            min_timestamp=interactions['timestamp'].min()
            #加上6个月的时间
            max_timestamp=min_timestamp+pd.DateOffset(months=12)
            #从interactions中取出时间在min_timestamp和max_timestamp之间的数据 
            interactions = interactions[(interactions['timestamp']>=min_timestamp) & (interactions['timestamp']<=max_timestamp)]
        elif args.sample_size=='36months':
            #从最小的timestamp开始，从中取出36个月的数据
            min_timestamp=interactions['timestamp'].min()
            #加上6个月的时间
            max_timestamp=min_timestamp+pd.DateOffset(months=36)
            #从interactions中取出时间在min_timestamp和max_timestamp之间的数据 
            interactions = interactions[(interactions['timestamp']>=min_timestamp) & (interactions['timestamp']<=max_timestamp)]
            
        elif args.sample_size=='500w':
            #从interactions中随机提取500w条数据
            interactions = interactions.sample(n=5000000)
        elif args.sample_size=='original':
            pass
        
        # Merge data,将interactions和itemfeat合并，合并的列是item_id
        data = pd.merge(interactions, users, on='userid')
        #按照时间戳排序
        data = data.sort_values(by='timestamp')
        #重置索引
        data.reset_index(drop=True, inplace=True)
        columns=data.columns.tolist()
        return data, columns
    
    
    elif dataset=='douban':
        dataset_path='/home/yunxshi/Data/douban'
        itemfeat_file=os.path.join(dataset_path, 'item_feat.csv')
        #itemfeat_file没有表头，需要自己添加，分隔符为空格
        itemfeat = pd.read_csv(itemfeat_file,sep=' ',header=None)

        #构造一个column，名字是feat，对应的值是每一列的值如果为1，就是对应的列名，用|分隔
        itemfeat['feat']=itemfeat.apply(lambda x: '|'.join([str(i) for i in x.index if x[i]==1]),axis=1)
        #只保留feat列
        itemfeat=itemfeat[['feat']]
        #增加一列item_id，从0开始
        itemfeat['item_id']=itemfeat.index

        interaction_file = os.path.join(dataset_path, 'ratings.csv')
        #interaction_file没有表头，需要自己添加，列名为['user_id','item_id','rating','timestamp','year']，分隔符为\t
        interactions = pd.read_csv(interaction_file,header=None,sep='\t',names=['user_id','item_id','rating','timestamp','year'])
        
        if args.task=='binary':
            #将rating为4,5的值转换为1，其他为0
            interactions['rating'] = interactions['rating'].apply(lambda x: 1 if x>=4 else 0)
        
        # Merge data,将interactions和itemfeat合并，合并的列是item_id
        data = pd.merge(interactions, itemfeat, on='item_id')
        #按照时间戳排序
        data = data.sort_values(by='timestamp')
        #重置索引
        data.reset_index(drop=True, inplace=True)
        columns=data.columns.tolist()
        return data, columns


def ana_dataset(dataset='ml-1m'):
    data, columns = get_merged_data(dataset)

if __name__ == "__main__":
    data, columns = get_merged_data('taobao2014')
    #统计有多少个不同的user_id, movie_id
    
    print('user_id的个数：',data['user_id'].nunique())
    print('movie_id的个数：',data['item_id'].nunique())
    #统计交互数据量
    print('交互数据量：',data.shape[0])
    #如果timestamp列的数据类型是int
    if data['time'].dtype==np.int64:
        print('最小时间：',datetime.fromtimestamp(data['time'].min()).strftime('%Y-%m-%d'))
        #最大时间，用年月日的格式表示
        print('最大时间：',datetime.fromtimestamp(data['time'].max()).strftime('%Y-%m-%d'))
    else:
        print('最小时间：',data['time'].min().strftime('%Y-%m-%d'))
        #最大时间，用年月日的格式表示
        print('最大时间：',data['time'].max().strftime('%Y-%m-%d'))




