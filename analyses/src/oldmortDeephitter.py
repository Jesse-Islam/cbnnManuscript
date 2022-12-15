import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv




def fitDeephit(train,fullTest,bsize,epochs,valida,ti='time',ev='status',patience=10,min_delta=10**-7,drpt=0.5,lay1=200,lay2=20):#,epoch,bsize,tPos,ePos):
  n= np.shape(train)[0]
  vl_size = int(n*0.20)
  df_val = valida#train.sample(frac=0.2)
  df_train = train#.drop(df_val.index)
  df_test = fullTest
  varNames = list(df_train.columns)
  unwanted = {ti, ev}
  varNames = [e for e in varNames if e not in unwanted]
  leave = [(col, None) for col in varNames]

  x_mapper = DataFrameMapper(leave)

  x_train = x_mapper.fit_transform(df_train).astype('float32')
  x_val = x_mapper.transform(df_val).astype('float32')
  x_test = x_mapper.transform(df_test).astype('float32')
 



  ########
  #prepare labels
  #######
  num_durations = 10
  labtrans = DeepHitSingle.label_transform(num_durations)
  get_target = lambda df: (df[ti].values, df[ev].values)
  y_train = labtrans.fit_transform(*get_target(df_train))
  y_val = labtrans.transform(*get_target(df_val))
  train = (x_train, y_train)
  val = (x_val, y_val)
  durations_test, events_test = get_target(df_test)
  ########
  #prepare model
  ########
  in_features = x_train.shape[1]
  num_nodes =[int(lay1),int(lay2)]
  out_features = labtrans.out_features
  batch_norm = True
  dropout = drpt
  
  #net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,activation=torch.nn.SELU)
  net = torch.nn.Sequential(
     torch.nn.BatchNorm1d(in_features),
     torch.nn.Linear(in_features,int(lay1)),
     #torch.nn.BatchNorm1d(int(lay1)),
     torch.nn.Tanh(),
     torch.nn.Dropout(float(dropout)),

     torch.nn.Linear(int(lay1), int(lay2)),
     #torch.nn.BatchNorm1d(int(lay2)),
     torch.nn.Tanh(),
     torch.nn.Dropout(float(dropout)),

     torch.nn.Linear(int(lay2), out_features))

  
  model = DeepHitSingle(net, tt.optim.Adam, alpha=0.5, sigma=0.1, duration_index=labtrans.cuts)
  lr_finder = model.lr_finder(x_train, y_train, batch_size=int(bsize), tolerance=2)
  #_ = lr_finder.plot()
  ############
  #fit models
  ###########
  model.optimizer.set_lr(0.0001)
  callbacks = [tt.callbacks.EarlyStopping(min_delta=min_delta, patience=patience)]
  log = model.fit(x_train, y_train, int(bsize),int(epochs), callbacks, val_data=val,verbose=False )
  return model.predict_surv_df(x_test)#log#[type(x_train), type(y_train[1])]
  
