# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com
    zanshuxun, zanshuxun@aliyun.com

"""
from __future__ import print_function
import copy

import time
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm

from functorch import make_functional_with_buffers, vmap, grad
import torch.nn.functional as F

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

from ..inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
from ..layers import PredictionLayer
from ..layers.utils import slice_arrays
from ..callbacks import History

feature_sum_linear = None
feature_sum_dnn = None
feature_dnn_per_batch=None
feature_linear_per_batch=None

def hook_linear(module, input, output):
    X=input[0]
    #将X转为二维的
    X=X.view(X.shape[0],-1)
    global feature_sum_linear
    if feature_sum_linear is None:
        feature_sum_linear=torch.sum(X, dim=0).detach()
    else:
        feature_sum_linear = torch.add(feature_sum_linear,torch.sum(X, dim=0).detach())
    return None

def hook_dnn(module, input, output):
    #input[0]: torch.Size([10000, 128])
    global feature_sum_dnn
    if feature_sum_dnn is None:
        feature_sum_dnn=torch.sum(input[0], dim=0).detach()
    else:
        feature_sum_dnn = torch.add(feature_sum_dnn,torch.sum(input[0], dim=0).detach())
    return None

def hook_dnn_per_batch(module, input, output):
    #input[0]: torch.Size([10000, 128])
    global feature_dnn_per_batch
    feature_dnn_per_batch=input[0].detach()
    return None

def hook_linear_per_batch(module, input, output):
    #input[0]: 
    global feature_linear_per_batch
    X=input[0].detach()
    feature_linear_per_batch=X.view(X.shape[0],-1)
    return None
   
class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):   
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]


        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        if len(sparse_embedding_list) > 0:
            #torch.Size([1000, 1, 7])
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            try:     
                linear_logit += sparse_feat_logit
            except:
                linear_logit=sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit
        return linear_logit


class BaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):

        super(BaseModel, self).__init__()
        self.model_dict={}
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()

    def get_lossInc(self, x=None, y=None,w1=None,w2=None):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
     
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))

        batch_size = 50000
        model = self.train()
        loss_func = self.loss_func
        optim = self.optim
        
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=False, batch_size=batch_size)
        
        #when w1
        loss_t=None
        model.load_state_dict(w1)
        with tqdm(enumerate(train_loader)) as t:
            for _, (x_train, y_train) in t:
                x = x_train.to(self.device).float()
                y = y_train.to(self.device).float()
                y_pred = model(x).squeeze()
                loss = loss_func(y_pred, y.squeeze(), reduction='none')
                if loss_t is None:
                    loss_t=loss.detach()
                else:
                    loss_t=torch.cat((loss_t,loss.detach()),0)

        #when w2
        loss_t1=None
        model.load_state_dict(w2)
        with tqdm(enumerate(train_loader)) as t:
            for _, (x_train, y_train) in t:
                x = x_train.to(self.device).float()
                y = y_train.to(self.device).float()
                y_pred = model(x).squeeze()
                loss = loss_func(y_pred, y.squeeze(), reduction='none')
                if loss_t1 is None:
                    loss_t1=loss.detach()
                else:
                    loss_t1=torch.cat((loss_t1,loss.detach()),0)
        lossInc=loss_t1-loss_t
        lossInc=lossInc.reshape(-1).detach().cpu().numpy()
        lossInc_dict = dict(zip(range(1,len(lossInc)+1), lossInc))
        return lossInc_dict
        
    def calculate_logitgap(self, x=None, y=None):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
     
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))

        batch_size = 50000

        model = self.eval()
        
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=False, batch_size=batch_size)
               
        logitgap=[]
        with tqdm(enumerate(train_loader)) as t:
            for _, (x_train, y_train) in t:
                x = x_train.to(self.device).float()
                y = y_train.to(self.device).float()
                y_pred = model(x)
                logitgap_tmp = (y_pred-y).reshape(-1).detach().cpu().numpy().tolist()
                logitgap.append(logitgap_tmp)
        logitgap_dict = dict(zip(range(1,len(logitgap)+1), logitgap))
        return logitgap_dict
    
    def get_feature(self, x=None, y=None):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
     
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))

        batch_size = 50000

        model = self.eval()
        
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=False, batch_size=batch_size)
   
        global feature_dnn_per_batch,feature_linear_per_batch
        features=None
        handle_dnn_linear_per=model.dnn_linear.register_forward_hook(hook_dnn_per_batch)
        hanle_linear_model_per=model.linear_model.register_forward_hook(hook_linear_per_batch)        
        
        #L2 distance of feature 
        for batch_idx, (x_train, y_train) in tqdm(enumerate(train_loader)):
            x = x_train.to(self.device).float()            
            model(x)
            feature= torch.cat((feature_dnn_per_batch,feature_linear_per_batch),1)
            feature=feature.reshape(len(y_train),-1).detach().cpu().numpy().tolist()
            if features is None:
                features=feature
            else:
                features.extend(feature)
                
            
            
        handle_dnn_linear_per.remove()
        hanle_linear_model_per.remove()
        feature_dnn_per_batch=None
        feature_linear_per_batch=None
          
        return np.array(features).reshape(len(y),-1)
    
        
    def get_featureL2(self, x=None, y=None):
    
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
     
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))

        
        batch_size = 50000

        model = self.eval()
        
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=False, batch_size=batch_size)

                    
        handle_dnn_linear=model.dnn_linear.register_forward_hook(hook_dnn)
        hanle_linear_model=model.linear_model.register_forward_hook(hook_linear)
        with tqdm(enumerate(train_loader)) as t:
            for _, (x_train, y_train) in t:
                x = x_train.to(self.device).float()
                y = y_train.to(self.device).float()

                model(x)
        global feature_sum_dnn,feature_sum_linear
        feature_mean_dnn=feature_sum_dnn/len(y)
        feature_mean_linear=feature_sum_linear/len(y)
        
        handle_dnn_linear.remove()
        hanle_linear_model.remove()
        feature_sum_linear = None
        feature_sum_dnn = None
        
        global feature_dnn_per_batch,feature_linear_per_batch
        L2=[]
        handle_dnn_linear_per=model.dnn_linear.register_forward_hook(hook_dnn_per_batch)
        hanle_linear_model_per=model.linear_model.register_forward_hook(hook_linear_per_batch)        
        
        for batch_idx, (x_train, y_train) in tqdm(enumerate(train_loader)):
            x = x_train.to(self.device).float()
            y = y_train.to(self.device).float()
            
            model(x)
            L2_dnn_tmp=torch.norm(feature_dnn_per_batch-feature_mean_dnn,p=2,dim=1)
            L2_linear_tmp=torch.norm(feature_linear_per_batch-feature_mean_linear,p=2,dim=1)
            L2.extend((L2_dnn_tmp+L2_linear_tmp).reshape(-1).cpu().detach().numpy().tolist())
            
        handle_dnn_linear_per.remove()
        hanle_linear_model_per.remove()
        feature_dnn_per_batch=None
        feature_linear_per_batch=None
                
        L2_dict = dict(zip(range(1,len(L2)+1), L2))
        
        return L2_dict
    
    def getCheckpoint(self,i):
        checkpoint_dir=os.path.join(self.args.root_path,'TrainPhase_%s'%self.args.dataset)
        pth_path=os.path.join(checkpoint_dir,'phase_%s.pth'%(i))
        return torch.load(pth_path)
    
    def getGrad(self,loss=None,gradient_fp=False):
        
        if loss is None:
            pass
        else:
            self.optim.zero_grad()
            loss.backward(retain_graph=True)  
        p_grad_list=[]
        model=self.eval()
        
        if gradient_fp:
            for param in model.parameters():
                p_grad=param.grad.reshape(-1)
                p_grad_list.append(p_grad)
        else:  
            for name, param in model.named_parameters():
                if name in ['dnn_linear.weight','linear_model.weight']:
                    p_grad=param.grad.reshape(-1)
                    p_grad_list.append(p_grad)
               
        grad=torch.cat(p_grad_list,-1).reshape(-1)
        return grad
    
    def calGGscoreAll(self,x2=None, y2=None,x1=None,y1=None,w_u=None,w_tao=None,dataset=None):
        
        if dataset in ['lastfm-1k','taobao2014']:
            batch_size = 10000
        else:
            batch_size = 10000
            
        if isinstance(x2, dict):
            x2 = [x2[feature] for feature in self.feature_index]
     
        for i in range(len(x2)):
            if len(x2[i].shape) == 1:
                x2[i] = np.expand_dims(x2[i], axis=1)

        union_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x2, axis=-1)),
            torch.from_numpy(y2))
        
        if isinstance(x1, dict):
            x1 = [x1[feature] for feature in self.feature_index]
     
        for i in range(len(x1)):
            if len(x1[i].shape) == 1:
                x1[i] = np.expand_dims(x1[i], axis=1)

        train_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x1, axis=-1)),
            torch.from_numpy(y1))
        
        union_data_loader = DataLoader(
            dataset=union_data, shuffle=False, batch_size=batch_size)
        train_data_loader=DataLoader(
            dataset=train_data, shuffle=False, batch_size=batch_size)
        

            
        loss_func = self.loss_func
        model = self.eval()
        class_name = model.__class__.__name__
        self.model_name=class_name
        
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        
        model.load_state_dict(w_u)
        grad_mean=None
        for batch_idx, (inputs, targets) in tqdm(enumerate(union_data_loader)):
            inputs, targets = inputs.to(self.device).float(), targets.to(self.device).float()
            outputs = model(inputs)
            tmp_loss = loss_func(outputs, targets)
            tmp_grad=self.getGrad(tmp_loss)
            if grad_mean is None:
                grad_mean=tmp_grad
            else:
                grad_mean+=tmp_grad
        grad_mean=grad_mean/len(union_data_loader)


        model.load_state_dict(w_tao)
        fmodel, params, buffers = make_functional_with_buffers(model) 


        def loss_fn(predictions, targets):
            return F.mse_loss(predictions, targets)
        
        def compute_loss_stateless_model (params_tograd,params, buffers, sample, target):
            for key, value in params_tograd.items():
                params[key]=value
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)

            predictions = fmodel(params, buffers, batch) 
            loss = loss_fn(predictions, targets)
            return loss
        
        ft_compute_grad = grad(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None,None,None, 0, 0))
        
        params_tograd={}
        
        if class_name in ['WDL','DeepFM']:
            if dataset=='lastfm-1k':
                key_list=[-1,4]
                for key in key_list:
                    params_tograd[key]=params[key]
            else:
                key_list=[-1]
                for key in key_list:
                    params_tograd[key]=params[key]
        elif class_name in ['DCN']:
            if dataset=='lastfm-1k':
                key_list=[-3,4]
                for key in key_list:
                    params_tograd[key]=params[key]
            else:
                key_list=[-3]
                for key in key_list:
                    params_tograd[key]=params[key]
            
        prod_all=[]
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_data_loader)):
            inputs, targets = inputs.to(self.device).float(), targets.to(self.device).float()
            
            ft_per_sample_grads = ft_compute_sample_grad(params_tograd,[p for p in params], buffers, inputs, targets)
   
            if dataset!='lastfm-1k':
                key = key_list[0]
                params_grads=ft_per_sample_grads[key].reshape(ft_per_sample_grads[key].shape[0] ,-1)
            elif dataset=='lastfm-1k':
                grad_list=[]
                for key in key_list:
                    grad_list.append(ft_per_sample_grads[key].reshape(ft_per_sample_grads[key].shape[0] ,-1))
                params_grads=torch.cat(grad_list,-1) 
                
            prod=torch.mm(params_grads,grad_mean.unsqueeze(1)).squeeze().detach().to('cpu').numpy()
            prod_all.extend(prod)         
        return dict(zip(range(1,len(prod_all)+1), prod_all))

    
    def calLossChangeAll(self,x=None, y=None,w_u=None,w_tao=None,dataset=None,gradient_fp=False):
        if gradient_fp:
            if dataset in ['lastfm-1k','taobao2014']:
                batch_size = 50
            else:
                batch_size = 500
        else:
            if dataset in ['lastfm-1k','taobao2014']:
                batch_size = 10000
            else:
                batch_size = 10000
            
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
     
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
            
        
        loss_func = self.loss_func
        model = self.eval()
        class_name = model.__class__.__name__
        self.model_name=class_name
        
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=False, batch_size=batch_size)
        
        model.load_state_dict(w_u)
        
        grad_mean=None
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
            inputs, targets = inputs.to(self.device).float(), targets.to(self.device).float()
            outputs = model(inputs)
            tmp_loss = loss_func(outputs, targets)
            tmp_grad=self.getGrad(tmp_loss,gradient_fp)
            if grad_mean is None:
                grad_mean=tmp_grad
            else:
                grad_mean+=tmp_grad
        grad_mean=grad_mean/len(train_loader)
        
        
        model.load_state_dict(w_tao)
        fmodel, params, buffers = make_functional_with_buffers(model)
        
        def loss_fn(predictions, targets):
            return F.mse_loss(predictions, targets)
        
        def compute_loss_stateless_model (params_tograd,params, buffers, sample, target):
            for key, value in params_tograd.items():
                params[key]=value
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)

            predictions = fmodel(params, buffers, batch) 
            loss = loss_fn(predictions, targets)
            return loss
        
        ft_compute_grad = grad(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None,None,None, 0, 0))
        
        params_tograd={}
            
        if gradient_fp:
            key_list=list(range(len(params)))
            for key in key_list:
                params_tograd[key]=params[key]
        else:
            if class_name in ['WDL','DeepFM']:
                if dataset=='lastfm-1k':
                    key_list=[-1,4]
                    for key in key_list:
                        params_tograd[key]=params[key]
                else:
                    key_list=[-1]
                    for key in key_list:
                        params_tograd[key]=params[key]
            elif class_name in ['NFM']:
                if dataset=='lastfm-1k':
                    key_list=[-1,4]
                    for key in key_list:
                        params_tograd[key]=params[key]
                else:
                    key_list=[-1]
                    for key in key_list:
                        params_tograd[key]=params[key]
            elif class_name in ['AFN']:
                if dataset=='lastfm-1k':
                    pass
                else:
                    key_list=[-2]
                    for key in key_list:
                        params_tograd[key]=params[key]
                        
            elif class_name in ['DCN']:
                if dataset=='lastfm-1k':
                    key_list=[-3,4]
                    for key in key_list:
                        params_tograd[key]=params[key]
                else :
                    key_list=[-3]
                    for key in key_list:
                        params_tograd[key]=params[key]
            
        prod_all=[]
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
            inputs, targets = inputs.to(self.device).float(), targets.to(self.device).float()
            ft_per_sample_grads = ft_compute_sample_grad(params_tograd,[p for p in params], buffers, inputs, targets)

            if gradient_fp:
                grad_list=[]
                for key in key_list:
                    grad_list.append(ft_per_sample_grads[key].reshape(ft_per_sample_grads[key].shape[0] ,-1))
                params_grads=torch.cat(grad_list,-1) 
            else:
                if dataset!='lastfm-1k':
                    key = key_list[0]
                    params_grads=ft_per_sample_grads[key].reshape(ft_per_sample_grads[key].shape[0] ,-1)
                elif dataset=='lastfm-1k':
                    grad_list=[]
                    for key in key_list:
                        grad_list.append(ft_per_sample_grads[key].reshape(ft_per_sample_grads[key].shape[0] ,-1))
                    params_grads=torch.cat(grad_list,-1) 
                
            prod=torch.mm(params_grads,grad_mean.unsqueeze(1)).squeeze().detach().to('cpu').numpy()
            prod_all.extend(prod)         
        return dict(zip(range(1,len(prod_all)+1), prod_all))
    
    def fit_kd(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,validation_data=None, shuffle=True, callbacks=None):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)


        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))

        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        #model copy
        old_model=copy.deepcopy(model)
            
        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:  
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    batch_index_new=0
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        if isinstance(loss_func, list):
                            assert len(loss_func) == self.num_tasks,\
                                "the length of `loss_func` should be equal with `self.num_tasks`"
                            loss = sum(
                                [loss_func[i](y_pred[:, i], y[:, i], reduction='sum') for i in range(self.num_tasks)])
                        else:
                            loss = loss_func(y_pred, y.squeeze(), reduction='sum')

                        y_pred_old=old_model(x).squeeze()
                        kd_loss=loss_func(y_pred, y_pred_old, reduction='sum')
                        
                        reg_loss = self.get_regularization_loss()

                        total_loss = kd_loss + loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

                        batch_index_new+=1
                                    


            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            
            # save model
            self.model_dict[epoch] =  copy.deepcopy(model.state_dict())

            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history
    
    
    
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,validation_data=None, shuffle=True, callbacks=None):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)


        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))

        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        if isinstance(loss_func, list):
                            assert len(loss_func) == self.num_tasks,\
                                "the length of `loss_func` should be equal with `self.num_tasks`"
                            loss = sum(
                                [loss_func[i](y_pred[:, i], y[:, i], reduction='sum') for i in range(self.num_tasks)])
                        else:
                            loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        #aa=input()
                        total_loss.backward()
                        #aa=input()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

                                    


            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            
            # save model
            #model.state_dict() 参数copy出来
            self.model_dict[epoch] =  copy.deepcopy(model.state_dict())

            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")
  
        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.001)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(),lr=0.001)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters(),lr=0.01)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters(),lr=0.01)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            loss_func = self._get_loss_func_single(loss)
        elif isinstance(loss, list):
            loss_func = [self._get_loss_func_single(loss_single) for loss_single in loss]
        else:
            loss_func = loss
        return loss_func

    def _get_loss_func_single(self, loss):
        if loss == "binary_crossentropy":
            loss_func = F.binary_cross_entropy
        elif loss == "mse":
            loss_func = F.mse_loss
        elif loss == "mae":
            loss_func = F.l1_loss
        else:
            raise NotImplementedError
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "rmse":
                    metrics_[metric] = root_mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = self._accuracy_score
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        # used for EarlyStopping in tf1.15
        return None

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]
