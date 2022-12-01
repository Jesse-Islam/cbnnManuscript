import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv



def fitDeepSurv(train,fullTest,bsize,epochs,ti='time',ev='status',patience=10,min_delta=10**-7,drpt=0.5):#,epoch,bsize,tPos,ePos):
  n= np.shape(train)[0]
  vl_size = int(n*0.20)
  df_val = train.sample(frac=0.2)
  df_train = train.drop(df_val.index)
  df_test = fullTest
  varNames = list(df_train.columns)
  unwanted = {ti, ev}
  varNames = [e for e in varNames if e not in unwanted]
  leave = [(col, None) for col in varNames]

  x_mapper = DataFrameMapper(leave)

  x_train = x_mapper.fit_transform(df_train).astype('float32')
  x_val = x_mapper.transform(df_val).astype('float32')
  x_test = x_mapper.transform(df_test).astype('float32')
  get_target = lambda df: (df[ti].values, df[ev].values)
  y_train = get_target(df_train)
  y_val = get_target(df_val)
  durations_test, events_test = get_target(df_test)
  val = x_val, y_val
  
  
  in_features = x_train.shape[1]
  num_nodes = [200,20]
  out_features = 1
  #batch_norm = False
  dropout = drpt
  
  #net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,activation=torch.tanh)
  net = torch.nn.Sequential(
     torch.nn.BatchNorm1d(in_features),
     torch.nn.Linear(in_features,200),
     torch.nn.Tanh(),
     torch.nn.Dropout(0.1),
    
     torch.nn.Linear(200, 20),
     torch.nn.Tanh(),
     torch.nn.Dropout(0.1),
    
     torch.nn.Linear(20, out_features))
  
  model = CoxPH(net, tt.optim.Adam)
  
  model.optimizer.set_lr(0.001)
  callbacks = [tt.callbacks.EarlyStopping(min_delta=min_delta, patience=patience)]
  log = model.fit(x_train, y_train, int(bsize),int(epochs), callbacks, val_data=val,verbose=False )
  #a,b=model.training_data
  #print(b)
  _ = model.compute_baseline_hazards()
  return model.predict_surv_df(x_test)
  
  
  
  
  
  
