# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score,roc_auc_score
import datetime 
import time
import lightgbm as lgb
import xgboost as xgb 
import warnings
import os
import gc

DATA_PATH='./data'
train_sales=pd.read_csv(f'{DATA_PATH}/train_sales_data.csv')
train_search=pd.read_csv(f'{DATA_PATH}/train_search_data.csv')
train_user=pd.read_csv(f'{DATA_PATH}/train_user_reply_data.csv')
evaluation_public=pd.read_csv(f'{DATA_PATH}/evaluation_public.csv')
submit_example=pd.read_csv(f'{DATA_PATH}/submit_example.csv')

train_sales.head(10)
train_search.head(10)
train_user.head(10)

data=train_sales.merge(train_search,on=('adcode','model','regYear','regMonth'),how='inner')
data.head(10)

new=data[['province_x','province_y']].assign(FLAG=data.province_x!=data.province_y)

new.FLAG
new.loc[new['FLAG']==False]['FLAG'].count()  #43296
new.loc[new['FLAG']!=False]['FLAG'].count()  #0
       
# province = province_y
       
data=data.drop(['province_x','province_y'],axis=1)
data.head(10)

#adcode,model,bodyType
import copy

df=copy.copy(pd.get_dummies(data['adcode'], drop_first=True))
#df.head()
data=pd.concat([data,df],axis=1)
#data.head()

#one hot encoding
categoricals = ['model', 'adcode','bodyType']
for feature in categoricals:
    df = copy.copy(pd.get_dummies(data[feature], drop_first=True))
    data= pd.concat([data, df], axis=1)
    data.drop(feature,axis = 1,inplace=True)
print(data.head())

             
#每个province有60个车型，一共22个省，就是1320，预测未来4个月，所以用的4为一个滑动窗口
data.iloc[1320*20:1320*24,:].values

def to_supervised(data):
    x=data.iloc[1320*0:1320*20,:].values
    y=data.iloc[1320*4:1320*24,2].values  #4为一个滑动窗口,salesVolume
    return x,y


data_x,data_y=to_supervised(data)
print(data_x.shape)
print(data_y.shape)
                  
train_x,test_x=data_x[0:1320*16],data_x[1320*16:1320*20]
train_y,test_y=data_y[0:1320*16],data_y[1320*16:1320*20]

print(test_x)
print(test_y)
print(test_y.shape)

####1. Try xgboost#################

from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric
from sklearn.metrics import r2_score 
from hyperopt import STATUS_OK,STATUS_RUNNING, fmin, hp, tpe,space_eval, partial
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
 
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from numpy.random import seed 
import numpy as np
import xgboost as xgb
import pandas as pd
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt

print("---------DMatrix----------")
dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(test_x, label=test_y)
##训练参数
SEED = 314159265
VALID_SIZE = 0.25 

def model_run(params):
    print("starting...")
    print("Training with params: ")
    print(params)
    num_boost_round=int(params['n_estimators'])
    print("watchlist")
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    print("training...")
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, verbose_eval=True)
    print("Validating...")
    check = gbm.predict(xgb.DMatrix(test_x), ntree_limit=gbm.best_iteration+1)
    #ntree_limit 是你迭代的树，第几棵树。 一般是 ntree_limit=model.best_iteration
    
    print("explained_variance_score...")
    score = get_score (test_y, check)
    print("pr...")
    print('Check error value: {:.6f}'.format(score))
   ## print("Predict test set...")
   ## test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration+1)
    return {
        'loss': score,
        'status': STATUS_OK,
        'stats_running': STATUS_RUNNING
    } 
         

def optimize(
             #trials, 
             random_state=SEED):
  
    space = {
        'n_estimators': hp.quniform('n_estimators', 20, 60, 1),
        'eta': hp.quniform('eta', 0.02, 0.4, 0.02),
        'max_depth':  hp.choice('max_depth', np.arange(1, 20, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'rmse',
        'objective': 'reg:linear',
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': random_state
    }
    
    print("---------开始训练参数----------")
    best = fmin(model_run, space, algo=tpe.suggest, max_evals=2000)
    ##print("---------------"+best+"-----------") 
    best_params = space_eval(space, best)
    print("BEST PARAMETERS: " + str(best_params))
    return best_params


def get_score(pre,real):
    temp=[]
    pre_t=[]
    real_t=[]
    pre=pre.round().astype(int)
    
    for i in range(60):
        for j in range(4):
            pre_t.append(pre[1320*j+22*i:1320*j+22*(i+1)])  #every month 60 model *22 prov
            real_t.append(real[1320*j+22*i:1320*j+22*(i+1)])
        temp.append(((mean_squared_error(pre_t,real_t))**0.5)/np.mean(real_t))
    return sum(temp)/60
print("---------开始优化参数----------")
best_params=optimize()
#print(test_prediction)
print("---------优化完成----------")
print(best_params) 


##训练模型
print(best_params)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model_gbm = xgb.train(best_params, dtrain, 180, evals=watchlist,early_stopping_rounds=50,verbose_eval=True)
print("Predict test set...")
test_prediction = model_gbm.predict(xgb.DMatrix(data.iloc[1320*20:1320*24,:].values), ntree_limit=model_gbm.best_iteration+1)
print(test_prediction)
print(best_params)
print(test_prediction.shape)
test_prediction=test_prediction.round().astype(int)
f = open('car.txt', 'w')
total = 0
for id in range(1320*4):
    str1 =str(test_prediction[total])
    str1 += '\n'
    total += 1
    f.write(str1)
f.close()
print("持久化完成")
test_prediction1=model_gbm.predict(xgb.DMatrix(test_x), ntree_limit=model_gbm.best_iteration+1)
test_prediction1=test_prediction1.round().astype(int)
score =get_score(test_y, test_prediction1)
print(1-score) 


####2. Try fbprophet#################
#pip install --index-url https://pypi.douban.com/simple fbprophet
from fbprophet import Prophet
from tqdm import tqdm

data=data.assign(Date=lambda data:data['regYear'].astype('str')+'/'+data['regMonth'].astype('str')+'/1')
#data=pd.concat([data,nds],axis=1)

#new=data[['regYear','regMonth']].assign(ds=data.regYear.concat.data.regMonth)

data.columns


prophet_data = train_sales.reset_index()

#prophet_data["ds"] = train_sales["Date"]

prophet_data["ds"] = data['Date']
prophet_data["y"] = train_sales["salesVolume"]
prophet_data.describe() 

pred = []

from sklearn.metrics import mean_squared_error
import numpy as np


train_sales
new_data=train_sales.merge(train_search,on=('adcode','model','regYear','regMonth'),how='inner')
new_data.head(10)
new_data.shape

train_user.shape
new_data=new_data.merge(train_user,on=('model','regYear','regMonth'),how='left')
new_data.shape


##select 1 city and 1 model for fbprophet
province_name = "上海"
class_df = new_data[new_data.province_x.str.startswith(province_name)].reset_index(drop=True)
model = "3c974920a76ac9c1"
class_df = class_df[class_df.model.str.startswith(model)].reset_index(drop=True)


df = pd.DataFrame(class_df)
df=df.assign(Datetimes=lambda df:df['regYear'].astype('str')+'/'+df['regMonth'].astype('str') +'/01')
df.head(2) 

df['Datetime1'] = pd.to_datetime(df.Datetimes,format='%Y/%m/%d %H:%M') 
 
prophet_data = df.reset_index()
prophet_data["ds"] = df['Datetime1'] 
prophet_data["y"] = df["salesVolume"]
prophet_data.describe() 

test_split = int(len(prophet_data) * 0.8)
prophet_data[: (test_split + 1)]
test_data = prophet_data[test_split:]

pred = []
for i in tqdm(range(len(prophet_data) - test_split)):
    data_to_fit = prophet_data[: (test_split + i)]
    prophet_model = Prophet(interval_width=0.95)
    prophet_model.fit(data_to_fit)
    prophet_forecast = prophet_model.make_future_dataframe(periods=4, freq="m")  #predict the incoming 4 months
    prophet_forecast = prophet_model.predict(prophet_forecast)
    pred.append(prophet_forecast["yhat"].iloc[-1])
    
mse_prophet = mean_squared_error(test_data.y, pred)
print("RMSE for PROPHET {:.2f}".format(np.sqrt(mse_prophet)))

from fbprophet.diagnostics import performance_metrics
prophet_model.plot_components(prophet_forecast)
prophet_model.plot(prophet_forecast)


from fbprophet.plot import add_changepoints_to_plot
fig = prophet_model.plot(prophet_forecast)
a = add_changepoints_to_plot(fig.gca(), prophet_model, prophet_forecast)
