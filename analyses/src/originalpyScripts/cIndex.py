
  # ### Evaluation
  # 
  # We evaluate the performance of DSM in its discriminative ability (Time Dependent Concordance Index and Cumulative Dynamic AUC) as well as Brier Score.
from sksurv.metrics import concordance_index_ipcw
import numpy as np
#outrisk is the prediction at time t for the cumulative incidence
def cidx(et_train, et_test, out_risk, time): 
  #print(et_train,"\n\n", et_test,"\n\n", out_risk,"\n\n", time)
  #train= et_train
  #test=et_test
  #train= (np.array(et_train[:,0],dtype = bool),np.array(et_train[:,1],dtype = bool))
  #test= (np.array(et_test[:,0],dtype = bool),np.array(et_test[:,1],dtype = bool))
  train = np.array([(et_train[i,0], et_train[i,1]) for i in range(np.shape(et_train)[0])],
                   dtype = [('e', bool), ('t', float)])
  test = np.array([(et_test[i,0], et_test[i,1]) for i in range(np.shape(et_test)[0])],
                   dtype = [('e', bool), ('t', float)])
  #print(test)          
  #return 
  return concordance_index_ipcw(train, test, out_risk[:,0], time)[0]
  
  #et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
  #                 dtype = [('e', bool), ('t', float)])
  #et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
  #                 dtype = [('e', bool), ('t', float)])
  #et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))],
  #                 dtype = [('e', bool), ('t', float)])
  #for i, _ in enumerate(times):
  #    cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])
  #brs.append(brier_score(et_train, et_test, out_survival, times)[1])
  #roc_auc = []
  #for i, _ in enumerate(times):
  #    roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])
  #for horizon in enumerate(horizons):
  #    print(f"For {horizon[1]} quantile,")
  #    print("TD Concordance Index:", cis[horizon[0]])
  #    print("Brier Score:", brs[0][horizon[0]])
  #    print("ROC AUC ", roc_auc[horizon[0]][0], "\n")
