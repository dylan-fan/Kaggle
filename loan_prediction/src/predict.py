'''
Created on 2014/01/29
@author: dylan
'''
import numpy as  np
from sklearn.preprocessing import MinMaxScaler

def knn_predictor(knn_model,X):
    dst_array, idx_array = knn_model.kneighbors(X)
    
    preds = []
    
    for i in range(dst_array.shape[0]):        
        if dst_array[i][0] > 10:
            y = 0
        else:
            idx = idx_array[i][0]
            y = knn_model.sss_y[idx]
        
        preds.append(y)
    
    return preds
            
    
def postprocess_predict(preds, threshold = 5):
    print 'threshold:', threshold
    preds = np.asarray(preds)
    preds[np.where(preds < threshold)[0]] = 0  
    preds = np.floor(preds)    

    return preds


def predictor_normlization(preds):
    for i in range(len(preds)):
        if preds[i] < 0:
            preds[i] = 0
        if preds[i] > 100:
            preds[i] = 100
    
    preds_log2 = np.log2(preds + 1)
    
    preds_max = np.max(preds_log2)
    preds_min = np.min(preds_log2)
    
    preds_norm = [(e - preds_min) * 100/ (preds_max - preds_min) for e  in preds_log2]
    
    for i in range(len(preds_norm)):
        if preds_norm[i] <25:
            preds_norm[i] = 0.0
            
        
        preds_norm[i] = np.floor(preds_norm[i])
        
        if preds_norm[i] > 92:
            preds_norm[i] = 100
        
#         print preds_norm[i]
        
    
    return preds_norm
    
    
  
    
    
if __name__ == '__main__':  
   pass