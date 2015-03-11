'''
Created on 2014/01/28
@author: dylan

'''

from sklearn import metrics

from sklearn import  cross_validation 
import numpy as np
import com_stat
import predict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics.metrics import auc
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.neighbors import KNeighborsRegressor
import com_util
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

# import libfm_process

import data_io

SEED = 15

def cv_stat(y_train, y_cv):
    y_zero = np.sum([1 if item == 0 else 0 for item in y_train])           
        
    n = len(y_train)
    
    
    print 'cv-train are: ',n
    print 'cv-train with y = 0 are: ', y_zero
    print 'cv-train ratio with y = 0 are: ', y_zero/(n + 0.0)
    
    
    y_zero = np.sum([1 if item == 0 else 0 for item in y_cv])           
    
    n = len(y_cv)
    
    
    print 'cv-test are: ',n
    print 'cv-test with y = 0 are: ', y_zero
    print 'cv-test ratio with y = 0 are: ', y_zero/(n + 0.0)
    

def cv_loop(X, y, model, N = 5):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=0.3,
            random_state=i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:, 1]
        auc = metrics.roc_auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N
  

def regressor_cv(model, X, y, fig = 1,  K =5):
    mean_score = 0.
    mean_baseline_score = 0.0
    print 'begin cross validation...'
    for j in range(K):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size = 0.3,
            random_state = j*SEED)
        print 'cross validation %d,train... '%j
        
        cv_stat(y_train, y_cv)
        
        model.fit(X_train, y_train)
        r_square = model.score(X_train, y_train)
        print 'regression R^2 : ', r_square
#         y_train_preds = model.predict(X_train)
#         y_train_preds = predict.postprocess_predict(y_train_preds, 1)
#         
#         for i in range(len(y_train_preds)):
#             if y_train[i] != 0:
#                 print y_train_preds[i], y_train[i]
#                 
#         for i in range(len(y_train_preds)):
#             if y_train[i] == 0 and y_train_preds[i] != 0:
#                 print y_train_preds[i], y_train[i]
                
#         com_stat.train_test_target_dis(y_train, y_cv)        
        print 'cross validation %d,predict... '%j
        preds = model.predict(X_cv)
        
        threshold = 5
        threshold_n = 0
        
        for e in preds:
            if e > threshold:
                threshold_n += 1
            
        
        print 'preds threshold:> %d, sample size:%d, ratio: %f '%(threshold, threshold_n, threshold_n/ (len(y_cv) + 0.0)) 
        
#         com_stat.target_preds_and_true_threshold_stat(preds, y_cv, threshold)
        preds = predict.postprocess_predict(preds, threshold)

#         preds = predict.predictor_normlization(preds)
       
        
#         for i in range(len(preds)):
#             if y_cv[i] != 0:
#                 print preds[i], y_cv[i]
                 
        for i in range(len(preds)):
            if y_cv[i] == 0 and preds[i] != 0:
                print preds[i], y_cv[i]
        
        loss_score = metrics.mean_absolute_error(y_cv, preds)
        print 'cross validation %d: MAE: %f'%(j,loss_score)
        baseline_loss_score = metrics.mean_absolute_error(y_cv, [0]*len(y_cv))
        print 'cross validation  %d baseline: MAE: %f'%(j,baseline_loss_score)
        
        if fig:
            com_stat.target_preds_and_true_dis(preds, y_cv)
            com_stat.target_hist_dis(preds)
        
        mean_score += loss_score
        mean_baseline_score += baseline_loss_score
        

    mean_score /= K
    mean_baseline_score /= K
    print '%d cross validation avg MAE: %f'%(K,mean_score)
    print '%d cross validation avg baseline MAE: %f'%(K,mean_baseline_score)
    
    print 'end cross validation...'
    
    return mean_score

def libfm_cv(X,y,fig = 1, K = 5,threshold = 10):
    mean_score = 0.
    mean_baseline_score = 0.0

    
    print 'begin cross validation...'
    for j in range(K):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size = 0.3,
            random_state = j*SEED)
        print 'cross validation %d,train... '%j
        cv_stat(y_train, y_cv)
        train_file = data_io.get_paths()['svmlight_format_cvtrain']
        test_file = data_io.get_paths()['svmlight_format_cvtest']
        preds_file = data_io.get_paths()['libfm_preds']
        preds = libfm_process.get_predict_probs(X_train, y_train, X_cv, y_cv, train_file, test_file, preds_file)
        preds = predict.postprocess_predict(preds, threshold = threshold)
        
        loss_score = metrics.mean_absolute_error(y_cv, preds)
        print 'cross validation %d: MAE: %f'%(j,loss_score)
        baseline_loss_score = metrics.mean_absolute_error(y_cv, [0]*len(y_cv))
        print 'cross validation  %d baseline: MAE: %f'%(j,baseline_loss_score)
        
        if fig:
            com_stat.target_preds_and_true_dis(preds, y_cv)
            com_stat.target_hist_dis(preds)
        
        mean_score += loss_score
        mean_baseline_score += baseline_loss_score
        

    mean_score /= K
    mean_baseline_score /= K
    print '%d cross validation avg MAE: %f'%(K,mean_score)
    print '%d cross validation avg baseline MAE: %f'%(K,mean_baseline_score)
    
    print 'end cross validation...'
    
    return mean_score
        
        
        
def classify_and_regressor_cv(classify_and_regressor_predictor, X, y, fig= 0,K =5, threshold = 5):
    mean_score = 0.
    mean_baseline_score = 0.0
    
    mean_auc_score = 0.0
    
    mean_loss_default_score = 0.0
    
    print 'begin cross validation...'
    for j in range(K):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size = 0.3,
            random_state = j*SEED)
        print 'cross validation %d,train... '%j
        cv_stat(y_train, y_cv)
        
        clf_preds_proba, preds = classify_and_regressor_predictor(X_train, y_train, X_cv,threshold)
        y_cv_classify = [1 if item > 0 else 0 for item in y_cv]
        auc = metrics.roc_auc_score(y_cv_classify, clf_preds_proba)
         
        print "AUC (fold %d/%d): %f" % (j+1, K, auc)
         
        mean_auc_score += auc
        loss_score = metrics.mean_absolute_error(y_cv, preds)
        print 'cross validation %d: MAE: %f'%(j+1,loss_score)
        loss_default_index = np.where(y_cv > 0)[0]
        loss_default_score = metrics.mean_absolute_error(y_cv[loss_default_index], preds[loss_default_index])
        print 'cross validation %d loss default MAE: %f'%(j+1, loss_default_score)
        
#         baseline_long_tail_index = np.where(preds> 20)[0]
#         preds[baseline_long_tail_index] = y_cv[baseline_long_tail_index]
#         adj_loss_score = metrics.mean_absolute_error(y_cv, preds)
#         print 'cross validation %d long-tail baseline: MAE: %f'%(j+1, adj_loss_score)
        
        baseline_loss_score = metrics.mean_absolute_error(y_cv, [0]*len(y_cv))
        print 'cross validation  %d baseline: MAE: %f'%(j+1,baseline_loss_score)
        

        mean_score += loss_score
        mean_baseline_score += baseline_loss_score
        mean_loss_default_score += loss_default_score
        
        if fig :
            com_stat.target_preds_and_true_dis(preds, y_cv)
        

    mean_score /= K
    mean_baseline_score /= K
    mean_auc_score /= K
    mean_loss_default_score /= K
    print '%d cross validation avg auc: %f'%(K,mean_auc_score)
    print '%d cross validation avg MAE: %f'%(K,mean_score)
    print '%d cross validation avg loss default MAE: %f'%(K,mean_loss_default_score)
    print '%d cross validation avg baseline MAE: %f'%(K,mean_baseline_score)
    
    print 'end cross validation...'
        

        
        
             

if __name__ == '__main__':
    pass
