

import numpy as np

import random
from dsm import datasets
#x, t, e = datasets.load_dataset('SUPPORT')
#t=t/10000
from sklearn.model_selection import ParameterGrid
from dsm import DeepSurvivalMachines
#import numpy as np
#split data
#n = len(x)

#tr_size = int(n*0.70)
#vl_size = int(n*0.10)
#te_size = int(n*0.20)

#x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size+vl_size]
#t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size+vl_size]
#e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size+vl_size]




#x = r.data.iloc[:,:3].to_numpy()
#x
#t =np.asarray( r.data.iloc[:,3])
#t
#e =np.asarray( r.data.iloc[:,4])
#e

def dsmfitter(train,fullTest,bsize,epochs,tPos=3,ePos=4):
  tPos=int(tPos)
  ePos=int(ePos)
  train2=train.to_numpy()#@np.random.shuffle(train) 
  np.random.shuffle(train2) 
  n= np.shape(train2)[0]
  vl_size = int(n*0.80)
  #vls=random.sample(range(n),vl_size)
  
  t_train = np.asarray( train2[:vl_size,tPos])#,dtype=np.float32)
  e_train = np.asarray( train2[:vl_size,ePos]).astype(int)
  x_train = train2[:vl_size,:]#.to_numpy()
  x_train = np.delete(x_train, [tPos,ePos], 1)
  
  t_val = np.asarray( train2[-vl_size:,tPos])#,dtype=np.float32)
  e_val = np.asarray( train2[-vl_size:,ePos]).astype(int)
  x_val = train2[-vl_size:,:]#.to_numpy()
  x_val = np.delete(x_val, [tPos,ePos], 1)

  #print(np.shape(train2))
  #print(np.shape(x_train))
  #print(np.shape(x_val))
  test=fullTest.to_numpy()
  
  t_test = np.asarray( test[:,tPos])
  e_test = np.asarray( test[:,ePos]).astype(int)
  x_test = np.delete(test, [tPos,ePos], 1)
  #print(x_test)
  #print(t_test)
 # print(e_test)
  horizons = np.arange(0, 1, 0.01).tolist()#[.25,.5,.75]#
  #times = horizons #
  times= np.quantile(t_test[e_test==1], horizons).tolist()
  
  
  # ### Setting the parameter grid
  # Lets set up the parameter grid to tune hyper-parameters. We will tune the number of underlying survival distributions, 
  # ($K$), the distribution choices (Log-Normal or Weibull), the learning rate for the Adam optimizer between $1\times10^{-3}$ and $1\times10^{-4}$ and the number of hidden layers between $0, 1$ and $2$.
  
  #from sklearn.model_selection import ParameterGrid
  
  param_grid = {'k' : [4],
                'distribution' : ['Weibull'],#,'LogNormal',
                'learning_rate' : [ 0.001],
                'layers' : [ [50,50,25,25] ]
               }
  params = ParameterGrid(param_grid)
  
  
  # ### Model Training and Selection
  
  #from dsm import DeepSurvivalMachines
  models = []
  for param in params:
      model = DeepSurvivalMachines(k = param['k'],
                                   distribution = param['distribution'],
                                   layers = param['layers'])
      # The fit method is called to train the model
      
     # model.set_callbacks(callbacks)
      model.fit(x_train, t_train, e_train, iters = int(epochs), learning_rate = param['learning_rate'],val_data=(x_val, t_val, e_val),batch_size=int(bsize))
      models.append([[model.compute_nll(x_val, t_val, e_val), model]])#x_val, t_val, e_val where validation would go
  best_model = min(models)
  model = best_model[0][1]
  
  
  # ### Inference
  
  #print(models)
  #print(np.shape(test))
  #out_risk = model.predict_risk(x_test, times)
  out_survival = model.predict_survival(x_test, times)
  
                   
  return out_survival
  
  
  
