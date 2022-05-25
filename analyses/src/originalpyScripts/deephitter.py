import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv


def fitDeephit(train,fullTest,bsize,epochs,ti='time',ev='status'):#,epoch,bsize,tPos,ePos):
  #print(train)
  n= np.shape(train)[0]
  vl_size = int(n*0.20)
  df_val = train.sample(frac=0.2)
  df_train = train.drop(df_val.index)
  df_test = fullTest
  varNames = list(df_train.columns)
  unwanted = {ti, ev}
  varNames = [e for e in varNames if e not in unwanted]
  #print(varNames)
  leave = [(col, None) for col in varNames]

  x_mapper = DataFrameMapper(leave)
  #print(x_mapper)

  x_train = x_mapper.fit_transform(df_train).astype('float32')
  x_val = x_mapper.transform(df_val).astype('float32')
  x_test = x_mapper.transform(df_test).astype('float32')
  
  #x_train = df_train.drop(columns=['time','status'],inplace=False)
  #x_val = df_val.drop(columns=['time','status'],inplace=False)
  #x_test = df_test.drop(columns=['time','status'],inplace=False)
  #cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
  #cols_leave = ['x4', 'x5', 'x6', 'x7']

 



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
  num_nodes = [50,50,25,25]
  out_features = labtrans.out_features
  batch_norm = False
  dropout = 0.5
  
  net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
  model = DeepHitSingle(net, tt.optim.Adam, alpha=0.5, sigma=0.1, duration_index=labtrans.cuts)
  lr_finder = model.lr_finder(x_train, y_train, batch_size=int(bsize), tolerance=2)
  #_ = lr_finder.plot()
  ############
  #fit models
  ###########
  model.optimizer.set_lr(0.001)
  callbacks = [tt.callbacks.EarlyStopping(min_delta=10**-7, patience=10)]
  log = model.fit(x_train, y_train, int(bsize),int(epochs), callbacks, val_data=val,verbose=False )
  return model.predict_surv_df(x_test)#log#[type(x_train), type(y_train[1])]
  
  
