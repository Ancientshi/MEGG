import os
import sys
import copy
import time

import numpy as np
import pandas as pd
import umap.umap_ as umap



sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import random
from math import cos, pi, ceil
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from config import *
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models import WDL, NFM, DCN
from Data.workspace.MEGG.examples.preprocess import *
from sklearn.metrics import mean_squared_error,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR,
                                      ReduceLROnPlateau, StepLR, _LRScheduler)
from torch.optim.sgd import SGD
from torchsummary import summary
from tqdm import tqdm
from utils import init_params, progress_bar
from metric import *

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

key2index = {}
def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


class Trainer(object):
    
    def __init__(self) -> None:
        
        self.args=args
        self.args.INCBLOCK=self.args.BLOCKNUM-self.args.BASEBLOCK
        self.interacted_items_dict={}
        self.records={}
        self.spend_time_records={}
        self.seen_users={}
        self.seen_items={}
        self.dirpath=os.path.join(self.args.root_path,'records_%s_%s_%s/%s/%s'%(self.args.dataset,self.args.BASEBLOCK,self.args.INCBLOCK,self.args.net,self.args.seed))
    
    def load_data(self):
        data,columns=get_merged_data(dataset=self.args.dataset)
        self.data=data
        self.columns=columns
        
        if self.args.dataset=='ml-1m':
            self.sparse_features = ["movie_id", "user_id",
                            "gender", "age", "occupation", "zip", ]
            self.target = ['rating']
            self.useridfield='user_id'
            self.itemidfield='movie_id'
        elif self.args.dataset=='ml-20m':
            self.sparse_features = ["movieId", "userId"]
            self.target = ['rating']
            self.useridfield='userId'
            self.itemidfield='movieId'
            
        elif self.args.dataset=='taobao2014':
            self.sparse_features = ["user_id", "item_id","item_category"]
            self.target = ['behavior_type']
            self.useridfield='user_id'
            self.itemidfield='item_id'

        elif self.args.dataset=='douban':
            self.sparse_features = ["user_id", "item_id"]
            self.target = ['rating']
            self.useridfield='user_id'
            self.itemidfield='item_id'
             
        elif self.args.dataset=='lastfm-1k':
            #['userid', 'timestamp', 'artid', 'artname', 'traid', 'traname', 'month', 'year', 'day', 'gender', 'age', 'country', 'signup', 'signup_month', 'signup_year', 'signup_day']

            self.sparse_features = ["userid", "artid","traid","country"]
            self.dense_features=["age","gender","signup_month","signup_year","signup_day","month","year","day"]
            self.target = ['like']
            self.useridfield='userid'
            self.itemidfield='traid'
            
    def feature_process(self):
        # 1.Label Encoding for sparse features,and process sequence features
        self.lbe_dict={}
        for feat in self.sparse_features:
            lbe = LabelEncoder()
            self.lbe_dict[feat]=lbe
            self.data[feat] = lbe.fit_transform(self.data[feat])
        
        if self.args.dataset=='lastfm-1k':
            #sparse_features = ["userid", "artid","traid","country"]
            fixlen_feature_columns = [SparseFeat(feat, self.data[feat].nunique(), embedding_dim='auto') for feat in ['userid','country']]
            fixlen_feature_columns += [SparseFeat(feat, self.data[feat].nunique(), embedding_dim=64) for feat in ['artid','traid']]
            
        # 2.count #unique features for each sparse field and generate feature config for sequence feature
        dim= 'auto' if self.args.embedding_dim==0 else self.args.embedding_dim
        fixlen_feature_columns = [SparseFeat(feat, self.data[feat].nunique(), embedding_dim=dim) for feat in self.sparse_features]
        
        if self.args.dataset in ['ml-1m','ml-20m']:
            # preprocess the sequence feature
            genres_list = list(map(split, self.data['genres'].values))
            genres_length = np.array(list(map(len, genres_list)))
            max_len = max(genres_length)
            # Notice : padding=`post`
            self.genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post' )

            varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(key2index) + 1, embedding_dim='auto'), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature
            
            if self.args.net in ['NFM','AFN']:
                varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(key2index) + 1, embedding_dim=dim), maxlen=max_len, combiner='mean')]
        
        elif self.args.dataset=='douban':
            # preprocess the sequence feature
            genres_list = list(map(split, self.data['feat'].values))
            genres_length = np.array(list(map(len, genres_list)))
            max_len = max(genres_length)
            # Notice : padding=`post`
            self.genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post' )

            varlen_feature_columns = [VarLenSparseFeat(SparseFeat('feat', vocabulary_size=len(key2index) + 1, embedding_dim='auto'), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature
            
            if self.args.net in ['NFM','AFN']:
                varlen_feature_columns = [VarLenSparseFeat(SparseFeat('feat', vocabulary_size=len(key2index) + 1, embedding_dim=dim), maxlen=max_len, combiner='mean')]
        
        elif self.args.dataset=='lastfm-1k':
            mms = MinMaxScaler(feature_range=(0, 1))
            self.data[self.dense_features] = mms.fit_transform(self.data[self.dense_features])
            fixlen_feature_columns+= [DenseFeat(feat, 1, ) for feat in self.dense_features]
    
        if self.args.dataset in ['ml-1m','ml-20m','douban']:
            self.linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
            self.dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
        elif self.args.dataset in ['taobao2014','lastfm-1k']:
            self.linear_feature_columns = fixlen_feature_columns
            self.dnn_feature_columns = fixlen_feature_columns

        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)
        
    def train_test_split(self):
        if self.args.dataset in ['ml-1m','ml-20m']:
            # 3.generate input data for model
            self.data['genres'] = self.genres_list
            blocksize=int(len(self.data)/self.args.BLOCKNUM)
            self.blocksize=blocksize
            datablock_list=[self.data[:blocksize*self.args.BASEBLOCK]]+[self.data[blocksize* (self.args.BASEBLOCK+i):blocksize * (self.args.BASEBLOCK+i + 1)] for i in range(self.args.INCBLOCK)]
            model_input_list=[]
            for i in range(0,self.args.INCBLOCK+1):
                tmp_model_input = {name: datablock_list[i][name] for name in self.sparse_features}  #
                tmp_model_input["genres"] = datablock_list[i]["genres"]
                model_input_list.append(tmp_model_input)
            
        elif self.args.dataset=='douban':
            # 3.generate input data for model
            self.data['feat'] = self.genres_list
            blocksize=int(len(self.data)/self.args.BLOCKNUM)
            self.blocksize=blocksize
            datablock_list=[self.data[:blocksize*self.args.BASEBLOCK]]+[self.data[blocksize* (self.args.BASEBLOCK+i):blocksize * (self.args.BASEBLOCK+i + 1)] for i in range(self.args.INCBLOCK)]
            model_input_list=[]
            for i in range(0,self.args.INCBLOCK+1):
                tmp_model_input = {name: datablock_list[i][name] for name in self.sparse_features}  #
                tmp_model_input["feat"] = datablock_list[i]["feat"]
                model_input_list.append(tmp_model_input)
            
        elif self.args.dataset == 'taobao2014':
            blocksize=int(len(self.data)/self.args.BLOCKNUM)
            self.blocksize=blocksize
            datablock_list=[self.data[:blocksize*self.args.BASEBLOCK]]+[self.data[blocksize* (self.args.BASEBLOCK+i):blocksize * (self.args.BASEBLOCK+i + 1)] for i in range(self.args.INCBLOCK)]
            model_input_list=[]
            for i in range(0,self.args.INCBLOCK+1):
                tmp_model_input = {name: datablock_list[i][name] for name in self.sparse_features}  #
                model_input_list.append(tmp_model_input)

        elif self.args.dataset == 'lastfm-1k':
            blocksize=int(len(self.data)/self.args.BLOCKNUM)
            self.blocksize=blocksize
            datablock_list=[self.data[:blocksize*self.args.BASEBLOCK]]+[self.data[blocksize* (self.args.BASEBLOCK+i):blocksize * (self.args.BASEBLOCK+i + 1)] for i in range(self.args.INCBLOCK)]
            model_input_list=[]
            for i in range(0,self.args.INCBLOCK+1):
                tmp_model_input = {name: datablock_list[i][name] for name in (self.sparse_features+self.dense_features)}  #
                model_input_list.append(tmp_model_input)
                
        self.datablock_list=datablock_list
        self.model_input_list=model_input_list
        
    def set_model(self):
        # 4.Define Model,compile and train
        device = 'gpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:%s' % self.args.device
        if self.args.net=='WDL':
            model = WDL(self.linear_feature_columns, self.dnn_feature_columns, task=self.args.task, device=device,dnn_dropout=0.5,dnn_hidden_units=(128,256,128),l2_reg_linear=0.01,l2_reg_embedding=0.01,l2_reg_dnn=0.01,seed=self.args.seed,dnn_activation='relu')
        elif self.args.net=='NFM':
            model = NFM(self.linear_feature_columns, self.dnn_feature_columns, task=self.args.task, device=device,dnn_dropout=0.5,dnn_hidden_units=(128,256,128),l2_reg_linear=0.01,l2_reg_embedding=0.01,l2_reg_dnn=0.01,seed=self.args.seed,dnn_activation='relu')
        elif self.args.net=='DCN':
            model = DCN(self.linear_feature_columns, self.dnn_feature_columns, task=self.args.task, device=device,dnn_dropout=0.5,dnn_hidden_units=(128,256,128),l2_reg_linear=0.01,l2_reg_embedding=0.01,l2_reg_dnn=0.01,seed=self.args.seed,dnn_activation='relu')
            
        if self.args.task=='binary':
            model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'auc'], )
        elif self.args.task=='regression':
            model.compile("adam", "mse", metrics=['mse','rmse'], )
            
        self.model=model
        

    def train_phase(self):
        if self.args.method in ['kd','losschange_replay_with_kd'] and self.block_num!=0:
            self.history = self.model.fit_kd(self.baseblock['x'], self.baseblock['y'], batch_size=self.args.train_batch_size, epochs=self.args.epoch, verbose=1,validation_split=0)
        else: 
            self.history = self.model.fit(self.baseblock['x'], self.baseblock['y'], batch_size=self.args.train_batch_size, epochs=self.args.epoch, verbose=1,validation_split=0)

    def train_prepare_GGscore(self,x,y):
        self.GGscore_history=self.model.fit(x,y, batch_size=self.args.train_batch_size, epochs=1, verbose=1,validation_split=0)
        
    def test_phase(self):
        metric_dict={}
        for incblock in self.incblock_stack:
            incblock_id=incblock[0]
            incblock_x=incblock[1]['x']
            incblock_y=incblock[1]['y']
            
            pred_ans = self.model.predict(incblock_x, batch_size=self.args.test_batch_size)
            if self.args.task=='binary':
                rmse=round(roc_auc_score(incblock_y, pred_ans), 4)
                print("test on datablock %s without cold users and items, AUC: %s"%(incblock_id,rmse))
            elif self.args.task=='regression':
                rmse=round(mean_squared_error(incblock_y, pred_ans)**0.5, 4)
                print("test on datablock %s without cold users and items, RMSE: %s"%(incblock_id,rmse))
                if self.args.dataset == 'taobao2014':
                    incblock_y=np.where(incblock_y>=3,1,0)
                    pred_ans=pred_ans/4
                    pred_ans=np.where(pred_ans<0,0,pred_ans)
                    pred_ans=np.where(pred_ans>1,1,pred_ans) 
                else:
                    pred_ans=np.delete(pred_ans,np.where(incblock_y==3))
                    incblock_y=np.delete(incblock_y,np.where(incblock_y==3))
                    incblock_y=np.where(incblock_y>=4,1,0)
                    pred_ans=pred_ans/5
                    pred_ans=np.where(pred_ans<0,0,pred_ans)
                    pred_ans=np.where(pred_ans>1,1,pred_ans)
                auc=round(roc_auc_score(incblock_y, pred_ans), 4)
                print("test on datablock %s without cold users and items, AUC: %s"%(incblock_id,auc))
                    
            metric_dict[incblock_id]=[rmse,auc]
            
            if self.args.eva_future_one:
                break
            
        self.records[self.block_num]=metric_dict
    
    def run(self):
        self.load_data()
        self.feature_process()
        self.train_test_split()
        self.set_model()
        self.train_phase()
        self.test_phase()
    
    def calculate_GGscore(self,x2,y2,x1,y1):
        if self.args.method=='GGscore1':
            '''train on the union, gradient on the union dataset, prune on the base dataset'''
            self.dataset_prune_on=x1
            return self.model.calGGscoreAll(x2, y2,x1, y1,self.model.model_dict[0],self.model.model_dict[self.args.epoch-1],self.args.dataset)  
        elif self.args.method=='GGscore2':
            ''' train on the union, gradient on the union dataset, prune on the union dataset'''
            self.dataset_prune_on=x2
            return self.model.calGGscoreAll(x2, y2,x2, y2,self.model.model_dict[0],self.model.model_dict[self.args.epoch-1],self.args.dataset)  
        elif self.args.method=='GGscore3':
            '''gradient on the union dataset, prune on the base dataset'''
            self.dataset_prune_on=x1
            return self.model.calGGscoreAll(x2, y2,x1, y1,self.model.model_dict[self.args.epoch-1],self.model.model_dict[self.args.epoch-2],self.args.dataset)
        elif self.args.method=='GGscore4':
            '''gradient on the union dataset, prune on the union dataset'''
            self.dataset_prune_on=x2
            return self.model.calGGscoreAll(x2, y2,x2, y2,self.model.model_dict[self.args.epoch-1],self.model.model_dict[self.args.epoch-2],self.args.dataset)

    def calculate_losschange(self,number=1):
        if number==1:
            w2=self.model.model_dict[self.args.epoch-1]
            w1=self.model.model_dict[self.args.epoch-2]
        elif number==2:
            w2=self.model.model_dict[self.args.epoch-2]
            w1=self.model.model_dict[self.args.epoch-3]
        elif number==3:
            w2=self.model.model_dict[self.args.epoch-3]
            w1=self.model.model_dict[self.args.epoch-4]
        elif  number==4:
            w2=self.model.model_dict[self.args.epoch-1]
            w1=self.model.model_dict[self.args.epoch-3]
        elif  number==5:
            w2=self.model.model_dict[self.args.epoch-1]
            w1=self.model.model_dict[self.args.epoch-4]
        return self.model.calLossChangeAll(self.baseblock['x'], self.baseblock['y'],w2,w1,self.args.dataset,self.args.gradient_fp)
    
    def replay_strategy(self,baseblock_losschange,strategy='remain_sides'):
        
        baseblock_losschange_org=copy.deepcopy(baseblock_losschange)
        baseblock_losschange=sorted(baseblock_losschange.items(),key=lambda x:x[1])
        baseblock_id=(np.array(baseblock_losschange)[:,0].reshape(-1)-1).astype(int).tolist()

        if self.args.replay_ratio==-1:
            self.args.replay_ratio=1-1/self.args.BASEBLOCK
            self.args.replay_ratio=round(self.args.replay_ratio,2)
            remain_size=len(baseblock_losschange)-self.blocksize
        else:
            remain_size=int(len(baseblock_losschange)*self.args.replay_ratio)
         
        
        if strategy=='remain_sides':
            left=baseblock_id[:remain_size//2]
            right=baseblock_id[-remain_size//2:]
            baseblock_remainid=left+right 
        elif strategy=='remain_mid':
            baseblock_remainid=baseblock_id[self.blocksize//2:-self.blocksize//2]
        elif strategy=='remain_right':
            baseblock_remainid=baseblock_id[len(baseblock_id)-remain_size:]
        elif strategy=='remain_left':
            baseblock_remainid=baseblock_id[:remain_size]

        if self.args.method in ['GGscore2','GGscore4']:
            return baseblock_remainid
        elif self.args.method in ['GGscore1','GGscore3']:
            prunedblock_x=copy.deepcopy(self.baseblock['x'])
            for key,value in prunedblock_x.items():             
                prunedblock_x[key]=value.iloc[baseblock_remainid]
            prunedblock_y=self.baseblock['y'][baseblock_remainid]
            return {'x':prunedblock_x,'y':prunedblock_y}
        else:
            prunedblock_x=copy.deepcopy(self.baseblock['x'])
            for key,value in prunedblock_x.items():             
                prunedblock_x[key]=value.iloc[baseblock_remainid]
            prunedblock_y=self.baseblock['y'][baseblock_remainid]
            return {'x':prunedblock_x,'y':prunedblock_y}
                 
    def update_baseblock_incblock(self,mode='update'):
        if mode=='init':
            print('Initializing Baseblock and Incblock...')
            print('Baseblock size: %s'%self.args.BASEBLOCK)
            print('Incblock size: %s'%self.args.INCBLOCK)
            self.baseblock={'x':self.model_input_list[0],'y':self.datablock_list[0][self.target].values} 
            self.incblock_stack=[]
            for i in range(1,self.args.INCBLOCK+1):
                self.incblock_stack.append((i,{'x':self.model_input_list[i],'y':self.datablock_list[i][self.target].values}))
        elif mode=='update':
            print('Updating Baseblock and Incblock...')
            print('It is the %s-th updation.'%self.block_num)
            print('Updating method: %s'%self.args.method)
            print('Incblock size: %s'%len(self.incblock_stack))
            
            incblock=self.incblock_stack.pop(0)
            if len(self.incblock_stack)==0:
                return
            
            while not os.path.exists(self.dirpath):
                time.sleep(5)
                try:
                    os.makedirs(self.dirpath)
                except:
                    pass
                    
            incblock_data=incblock[1]
            if self.args.method=='full_batch':
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(self.baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    self.baseblock['x'][key]=incValue
                self.baseblock['y']=np.concatenate((self.baseblock['y'],incblock_data['y']),axis=0)
                #train from scratch
                self.set_model()
                
            elif self.args.method=='kd':
                baseblock_x=incblock_data['x']
                baseblock_y=incblock_data['y']
                self.baseblock={'x':baseblock_x,'y':baseblock_y} 
                
            elif self.args.method=='fine_tune':
                baseblock_x=incblock_data['x']
                baseblock_y=incblock_data['y']
                self.baseblock={'x':baseblock_x,'y':baseblock_y} 
                
            elif self.args.method in ['random','random_with_kd']:    
                if self.args.replay_ratio==-1:
                    self.args.replay_ratio=1-1/self.args.BASEBLOCK
                    self.args.replay_ratio=round(self.args.replay_ratio,2)
                    remain_size=len(self.baseblock['y'])-self.blocksize
                else:
                    remain_size=int(len(self.baseblock['y'])*self.args.replay_ratio)
                
                prunedblock_x=copy.deepcopy(self.baseblock['x'])
                baseblock_remainid=random.sample(range(0,len(self.baseblock['y'])),remain_size)
                for key,value in prunedblock_x.items():             
                    prunedblock_x[key] = value.iloc[baseblock_remainid]
                prunedblock_y=self.baseblock['y'][baseblock_remainid]
                pruned_baseblock={'x':prunedblock_x,'y':prunedblock_y}
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(pruned_baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    incValue.reset_index(drop=True,inplace=True)
                    self.baseblock['x'][key]=incValue
                self.baseblock['y']=np.concatenate((pruned_baseblock['y'],incblock_data['y']),axis=0)
                if self.args.method=='random':
                    #train from scratch
                    self.set_model()
                elif self.args.method=='random_with_kd':
                    pass
                
            elif self.args.method in ['mir','mir_with_kd']:
                lossInc_path=os.path.join(self.dirpath,'lossInc_baseblock_%s.npy'%self.block_num)
                
                baseblock_lossInc=self.model.get_lossInc(self.baseblock['x'], self.baseblock['y'],self.model.model_dict[self.args.epoch-1],self.model.model_dict[self.args.epoch-2])
                np.save(lossInc_path,baseblock_lossInc)
                
                pruned_baseblock=self.replay_strategy(baseblock_lossInc,'remain_right')
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(pruned_baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    incValue.reset_index(drop=True,inplace=True)
                    self.baseblock['x'][key]=incValue
                self.baseblock['y']=np.concatenate((pruned_baseblock['y'],incblock_data['y']),axis=0)
                if self.args.method=='mir':
                    #train from scratch
                    self.set_model()
                elif self.args.method=='mir_with_kd':
                    pass
                
            elif self.args.method in ['herding','herding_with_kd']:
                featureL2_path=os.path.join(self.dirpath,'featureL2_baseblock_%s.npy'%self.block_num)
                
                baseblock_featureL2=self.model.get_featureL2(self.baseblock['x'], self.baseblock['y'])
                np.save(featureL2_path,baseblock_featureL2)
                
                pruned_baseblock=self.replay_strategy(baseblock_featureL2,'remain_left')
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(pruned_baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    incValue.reset_index(drop=True,inplace=True)
                    self.baseblock['x'][key]=incValue
                self.baseblock['y']=np.concatenate((pruned_baseblock['y'],incblock_data['y']),axis=0)
                if self.args.method=='herding':
                    #train from scratch
                    self.set_model()
                elif self.args.method=='herding_with_kd':
                    pass
                
                
            elif self.args.method=='logitgap':
                logitgap_path=os.path.join(self.dirpath,'logitgap_baseblock_%s.npy'%self.block_num)
                if not os.path.exists(logitgap_path):
                    baseblock_logitgap=self.model.calculate_logitgap(self.baseblock['x'], self.baseblock['y'])
                    np.save(logitgap_path,baseblock_logitgap)
                else:
                    baseblock_logitgap=np.load(logitgap_path, allow_pickle=True).item()
                pruned_baseblock=self.replay_strategy(baseblock_logitgap,'remain_sides')
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(pruned_baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    incValue.reset_index(drop=True,inplace=True)
                    self.baseblock['x'][key]=incValue
                self.baseblock['y']=np.concatenate((pruned_baseblock['y'],incblock_data['y']),axis=0)
                #train from scratch
                self.set_model()
            
            
            elif self.args.method in ['losschange_replay','losschange_replay_with_kd']:
                fp_suffix='_fp' if self.args.gradient_fp else ''
                losschange_path=os.path.join(self.dirpath,'losschange_%s_baseblock_%s%s.npy'%(self.args.strategy,self.block_num,fp_suffix))
                
                baseblock_losschange=self.calculate_losschange()
                np.save(losschange_path,baseblock_losschange)
                
                pruned_baseblock=self.replay_strategy(baseblock_losschange,self.args.strategy)
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(pruned_baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    incValue.reset_index(drop=True,inplace=True)
                    self.baseblock['x'][key]=incValue
                self.baseblock['y']=np.concatenate((pruned_baseblock['y'],incblock_data['y']),axis=0)
                
                if self.args.method=='losschange_replay':
                    self.set_model()
                elif self.args.method=='losschange_replay_with_kd':
                    pass
                     
                
            elif self.args.method.startswith('losschange') and self.args.method.endswith('_replay') and self.args.method!='losschange_replay':
                number=int(self.args.method.split('_')[0].replace('losschange',''))

                fp_suffix='_fp' if self.args.gradient_fp else ''
                losschange_path=os.path.join(self.dirpath,'losschange%s_%s_baseblock_%s%s.npy'%(number,self.args.strategy,self.block_num,fp_suffix))
                
                
                baseblock_losschange=self.calculate_losschange(number=number)
                np.save(losschange_path,baseblock_losschange)
                    
                pruned_baseblock=self.replay_strategy(baseblock_losschange,self.args.strategy)
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(pruned_baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    incValue.reset_index(drop=True,inplace=True)
                    self.baseblock['x'][key]=incValue
                self.baseblock['y']=np.concatenate((pruned_baseblock['y'],incblock_data['y']),axis=0)
                self.set_model()
            
                
            elif self.args.method=='GGscore1':
                '''
                GGscore1: train on the union, gradient on the union dataset, prune on the base dataset
                '''
                GGscore1_path=os.path.join(self.dirpath,'GGscore1_%s_baseblock_%s.npy'%(self.args.strategy,self.block_num))
                
                
                baseblock=copy.deepcopy(self.baseblock)
                x1=copy.deepcopy(self.baseblock['x'])
                y1=copy.deepcopy(self.baseblock['y'])
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    #重置index
                    incValue.reset_index(drop=True,inplace=True)
                    baseblock['x'][key]=incValue
                baseblock['y']=np.concatenate((baseblock['y'],incblock_data['y']),axis=0)
                #train
                print("Train on the union dataset, make preparation for calculating GGscore.")
                self.train_prepare_GGscore(baseblock['x'],baseblock['y'])
                            
                baseblock_GGscore=self.calculate_GGscore(x2=baseblock['x'],y2=baseblock['y'],x1=x1,y1=y1)
                np.save(GGscore1_path,baseblock_GGscore)
                    
                    
                pruned_baseblock=self.replay_strategy(baseblock_GGscore,self.args.strategy)
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(pruned_baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    #重置index
                    incValue.reset_index(drop=True,inplace=True)
                    self.baseblock['x'][key]=incValue
                self.baseblock['y']=np.concatenate((pruned_baseblock['y'],incblock_data['y']),axis=0)
                self.set_model()
                
            elif self.args.method=='GGscore2':
                '''
                GGscore2: train on the union, gradient on the union dataset, prune on the union dataset
                '''
                GGscore2_path=os.path.join(self.dirpath,'GGscore2_%s_baseblock_%s.npy'%(self.args.strategy,self.block_num))
                
                baseblock=copy.deepcopy(self.baseblock)
                x1=copy.deepcopy(self.baseblock['x'])
                y1=copy.deepcopy(self.baseblock['y'])
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    #重置index
                    incValue.reset_index(drop=True,inplace=True)
                    baseblock['x'][key]=incValue
                baseblock['y']=np.concatenate((baseblock['y'],incblock_data['y']),axis=0)
                #train
                print("Train on the union dataset, make preparation for calculating GGscore.")
                self.train_prepare_GGscore(baseblock['x'],baseblock['y'])
                            
                baseblock_GGscore=self.calculate_GGscore(x2=baseblock['x'],y2=baseblock['y'],x1=x1,y1=y1)
                np.save(GGscore2_path,baseblock_GGscore)
                    
                    
                baseblock_remainid=self.replay_strategy(baseblock_GGscore,self.args.strategy)
                for key,value in incblock_data['x'].items():
                    unionValue=pd.concat([pd.DataFrame(self.baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    #从unionValue中选出baseblock_remainid对应的行
                    value=unionValue.iloc[baseblock_remainid]
                    value.reset_index(drop=True,inplace=True)
                    self.baseblock['x'][key]=value
                unionY=np.concatenate((self.baseblock['y'],incblock_data['y']),axis=0)
                self.baseblock['y']=unionY[baseblock_remainid]
                
                #train from scratch
                self.set_model()
            
            elif self.args.method=='GGscore3':
                '''
                GGscore3: gradient on the union dataset, prune on the base dataset
                '''
                GGscore3_path=os.path.join(self.dirpath,'GGscore3_%s_baseblock_%s.npy'%(self.args.strategy,self.block_num))
                
                baseblock=copy.deepcopy(self.baseblock)
                x1=copy.deepcopy(self.baseblock['x'])
                y1=copy.deepcopy(self.baseblock['y'])
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    #重置index
                    incValue.reset_index(drop=True,inplace=True)
                    baseblock['x'][key]=incValue
                baseblock['y']=np.concatenate((baseblock['y'],incblock_data['y']),axis=0)
                            
                baseblock_GGscore=self.calculate_GGscore(x2=baseblock['x'],y2=baseblock['y'],x1=x1,y1=y1)
                np.save(GGscore3_path,baseblock_GGscore)
                    
                    
                pruned_baseblock=self.replay_strategy(baseblock_GGscore,self.args.strategy)
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(pruned_baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    #重置index
                    incValue.reset_index(drop=True,inplace=True)
                    self.baseblock['x'][key]=incValue
                self.baseblock['y']=np.concatenate((pruned_baseblock['y'],incblock_data['y']),axis=0)
                self.set_model()
                
            elif self.args.method=='GGscore4':
                '''
                GGscore4:  gradient on the union dataset, prune on the union dataset
                '''
                GGscore4_path=os.path.join(self.dirpath,'GGscore4_%s_baseblock_%s.npy'%(self.args.strategy,self.block_num))
                
                baseblock=copy.deepcopy(self.baseblock)
                x1=copy.deepcopy(self.baseblock['x'])
                y1=copy.deepcopy(self.baseblock['y'])
                for key,value in incblock_data['x'].items():
                    incValue=pd.concat([pd.DataFrame(baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    #重置index
                    incValue.reset_index(drop=True,inplace=True)
                    baseblock['x'][key]=incValue
                baseblock['y']=np.concatenate((baseblock['y'],incblock_data['y']),axis=0)

                            
                baseblock_GGscore=self.calculate_GGscore(x2=baseblock['x'],y2=baseblock['y'],x1=x1,y1=y1)
                np.save(GGscore4_path,baseblock_GGscore)
                    
                         
                baseblock_remainid=self.replay_strategy(baseblock_GGscore,self.args.strategy)
                for key,value in incblock_data['x'].items():
                    unionValue=pd.concat([pd.DataFrame(self.baseblock['x'][key]),pd.DataFrame(value)],axis=0)
                    value=unionValue.iloc[baseblock_remainid]
                    value.reset_index(drop=True,inplace=True)
                    self.baseblock['x'][key]=value
                unionY=np.concatenate((self.baseblock['y'],incblock_data['y']),axis=0)
                self.baseblock['y']=unionY[baseblock_remainid]   
            
                
    def irs(self):
        if not os.path.exists(self.dirpath):
                os.makedirs(self.dirpath)
                
        self.load_data()
        self.feature_process()
        self.train_test_split()
        self.set_model()
        
        self.update_baseblock_incblock('init')
        
        self.block_num=0
        start_time=time.time()    
        while len(self.incblock_stack)>=1:
            print("Train on the baseblock, and calculate the LossChange of each samples.")
            if self.args.method=='full_batch':
                pth_path=os.path.join(self.dirpath,'''phase_%s_%s.pth'''%(self.args.method,self.block_num))
                self.train_phase()
                end_time=time.time()
                torch.save(self.model.state_dict(),pth_path)
                print("Predict on the incblock.")
                self.test_phase()
            else:
                self.train_phase() 
                end_time=time.time()   
                print("Predict on the incblock.")
                self.test_phase()
            time_cost=round(end_time-start_time,2)
            self.spend_time_records[self.block_num] = time_cost
              
            
            nocold_suffix='_nocold' if self.args.evaluation_nocold else ''
            fp_suffix='_fp' if self.args.gradient_fp else ''
            
            if self.args.method in ['losschange_replay','GGscore1','GGscore2','GGscore3','GGscore4','losschange_replay_with_kd']:
                records_path=os.path.join(self.dirpath,'records_%s_%s%s%s_%s.txt'%(self.args.method,self.args.strategy,nocold_suffix,fp_suffix,self.args.replay_ratio))
                time_records_path=os.path.join(self.dirpath,'time_records_%s_%s%s%s_%s.txt'%(self.args.method,self.args.strategy,nocold_suffix,fp_suffix,self.args.replay_ratio))
                
            elif self.args.method.startswith('losschange') and self.args.method.endswith('_replay') and self.args.method!='losschange_replay':
                records_path=os.path.join(self.dirpath,'records_%s_%s%s%s_%s.txt'%(self.args.method,self.args.strategy,nocold_suffix,fp_suffix,self.args.replay_ratio))
                time_records_path=os.path.join(self.dirpath,'time_records_%s_%s%s%s_%s.txt'%(self.args.method,self.args.strategy,nocold_suffix,fp_suffix,self.args.replay_ratio))
            elif self.args.method in ['herding','mir','random']:
                records_path=os.path.join(self.dirpath,'records_%s%s%s_%s.txt'%(self.args.method,nocold_suffix,fp_suffix,self.args.replay_ratio))
                time_records_path=os.path.join(self.dirpath,'time_records_%s%s%s_%s.txt'%(self.args.method,nocold_suffix,fp_suffix,self.args.replay_ratio))
            else:
                records_path=os.path.join(self.dirpath,'records_%s%s%s.txt'%(self.args.method,nocold_suffix,fp_suffix))
                time_records_path=os.path.join(self.dirpath,'time_records_%s%s%s.txt'%(self.args.method,nocold_suffix,fp_suffix))
            with open(records_path,'w') as f:
                f.write(str(self.records))
            with open(time_records_path,'w') as f:
                f.write(str(self.spend_time_records))
            
            start_time=time.time()
            self.update_baseblock_incblock('update')
            self.block_num+=1

if __name__ == "__main__":
    trainer = Trainer()
    trainer.irs()
