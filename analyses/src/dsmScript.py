import numpy as np
import random
#from dsm import datasets
from sklearn.model_selection import ParameterGrid
from dsm import DeepSurvivalMachines
##in dsm/utilities.py, we set early stopping patience and min_delta manually
##dsm/dsm_torch.py, line 89 dropout(0.5) layer was added


def dsmfitter(train,fullTest,bsize,epochs,val,times,tPos=3,ePos=4,lay1=50,lay2=50,lay3=25,lay4=25,lr=0.001):
  
  ###############################
  ###Prepare data
  ###############################
  tPos=int(tPos)
  ePos=int(ePos)
  train2=train.to_numpy()#@np.random.shuffle(train) 
  #np.random.shuffle(train2) 
  n= np.shape(train2)[0]
  #vl_size = int(n*0.80)

  
  t_train = np.asarray( train2[:,tPos])
  e_train = np.asarray( train2[:,ePos]).astype(int)
  x_train = train2
  x_train = np.delete(x_train, [tPos,ePos], 1)
  val=val.to_numpy()
  t_val = np.asarray( val[:,tPos])
  e_val = np.asarray( val[:,ePos]).astype(int)
  x_val = val
  x_val = np.delete(x_val, [tPos,ePos], 1)
  #print("hi")
  test=fullTest.to_numpy()
  t_test = np.asarray( test[:,tPos])
  e_test = np.asarray( test[:,ePos]).astype(int)
  x_test = np.delete(test, [tPos,ePos], 1)
  
  horizons = np.arange(0, 1, 0.01).tolist()
  times= np.quantile(t_test[e_test==1], horizons).tolist()
  
  

  ###############################
  ###Prepare data
  ###############################
  #I fix the parameter grid
  param_grid = {'k' : [6],
                'distribution' : ['Weibull'],
                'learning_rate' : [ lr],
                'layers' : [ [int(lay1),int(lay2)] ]
               }
  params = ParameterGrid(param_grid)
  
  #note that the DeepSurvivalMachines function has been modified: 
  #it now introduces a dropout layer after each feedforward dense layer.
  # ### Model Training and Selection
  models = []
  for param in params:
      model = DeepSurvivalMachines(k = param['k'],
                                   distribution = param['distribution'],
                                   layers = param['layers'])
      model.fit(x_train, t_train, e_train, iters = int(epochs), learning_rate = param['learning_rate'],val_data=(x_val, t_val, e_val),batch_size=int(bsize))
      models.append([[model.compute_nll(x_val, t_val, e_val), model]])#x_val, t_val, e_val where validation would go
  best_model = min(models)
  model = best_model[0][1]
  out_survival = model.predict_survival(x_test, times)
  return out_survival
  
  
  
