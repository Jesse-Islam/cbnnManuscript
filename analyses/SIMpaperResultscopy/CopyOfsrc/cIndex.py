#We evaluate the performance of DSM in its discriminative ability (Time Dependent Concordance Index and Cumulative Dynamic AUC) as well as Brier Score.
from sksurv.metrics import concordance_index_ipcw
import numpy as np
def cidx(et_train, et_test, out_risk): 
  train = np.array([(et_train[i,0], et_train[i,1]) for i in range(np.shape(et_train)[0])],
                   dtype = [('e', bool), ('t', float)])
  test = np.array([(et_test[i,0], et_test[i,1]) for i in range(np.shape(et_test)[0])],
                   dtype = [('e', bool), ('t', float)])
  return concordance_index_ipcw(train, test, out_risk[:,0])[0]

