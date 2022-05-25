#import dsm
#def getSupportDSM():
#  return dsm.datasets.load_dataset('SUPPORT')


import pycox
def getSupportPycox():
  return pycox.datasets.support.read_df()
