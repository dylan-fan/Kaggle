'''
Created on 2014/01/28
@author: dylan
'''

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsRegressor


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
import scipy as sp

import cv
import data_io
import feature_generator

import com_util

import predict

import numpy as np

import csv
import com_stat

from ensemble import ensemble_preditor
from ensemble import ensemble_regression_preditor
# import rpy2
# from rpy2 import robjects as ro
# from rpy2.robjects.packages import importr
import os

import adaptive_synthetic_sampling as adsys
    

import random
    
    
def long_tail_regressor_predictor(train_x, train_y, test_x, stage0_regressor_preds):
    print 'long-tail predictor....'
    long_tail_train_index = np.where(train_y > 10)[0]
    long_tail_train_x = train_x[long_tail_train_index]
    long_tail_train_y = train_y[long_tail_train_index]
    
    long_tail_test_index = np.where(stage0_regressor_preds > 20)[0]
    long_tail_test_x = test_x[long_tail_test_index]
    long_tail_train_x ,long_tail_train_y = com_stat.train_sample_boostrap(long_tail_train_x ,long_tail_train_y)
    
    regressor_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.3,
                               min_samples_split= 5,min_samples_leaf= 2,  loss='lad')
      
    regressor_model.fit(long_tail_train_x ,long_tail_train_y)
    preds = regressor_model.predict(long_tail_test_x)  
  
#     preds = Quantile_Regression(long_tail_train_x ,long_tail_train_y,long_tail_test_x)
    
    print 'stage0 long-tail preds:',stage0_regressor_preds[long_tail_test_index]
#     stage0_regressor_preds[long_tail_test_index] = (preds + stage0_regressor_preds[long_tail_test_index] )/ 2.0  
    stage0_regressor_preds[long_tail_test_index] = preds * 0.7 + stage0_regressor_preds[long_tail_test_index] * 0.3
#     ; mae = 0.510
#     stage0_regressor_preds[long_tail_test_index] = preds
    print 'stage1 long-tail preds:',stage0_regressor_preds[long_tail_test_index]
    
    return stage0_regressor_preds


def Quantile_Regression(train_x, train_y, test_x):
    train_y = np.asarray([train_y]).T    
    train_x = np.hstack((train_x,train_y))
    data_io.save_train_feature(train_x)
    data_io.save_test_feature(test_x)
    
    os.system('Rscript ../R/analysis.R')
    preds = data_io.read_quantile_regression_preds()
    return preds

def NonlinearRegression_CV(train_x_regressor, train_y_regressor, test_x_regressor, model_type = 'GBR'):
    
    if model_type == 'GBR':
        regressor_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                               min_samples_split=30,min_samples_leaf= 5, loss='lad')
    if model_type == 'ETR':
        regressor_model = ExtraTreesRegressor(n_estimators= 1500,
                                min_samples_split=15, min_samples_leaf= 5)
        
    
#     train_x_regressor, train_y_regressor = adsys.generateSamples(train_x_regressor, train_y_regressor)
    kfold = cross_validation.StratifiedKFold(train_y_regressor, 20)
    stack_preds = []
    
    for cv_train_idx, cv_test_idx in kfold:
        
        sample_weight = com_stat.stat_sample_weight(train_y_regressor[cv_train_idx])
    
        try:
            regressor_model.fit(train_x_regressor[cv_train_idx], train_y_regressor[cv_train_idx],sample_weight)
        except :
            print 'sample weight parameter is not legal...'
            regressor_model.fit(train_x_regressor[cv_train_idx], train_y_regressor[cv_train_idx])        
        
        
        test_y_regressor_preds = regressor_model.predict(test_x_regressor)
        stack_preds.append(test_y_regressor_preds)
    
    test_y_regressor_preds = np.median(np.array(stack_preds).T,axis=1)
        

    
    # stage1: first regressor for all samples;   
    
    
    return test_y_regressor_preds
    
def NonlinearRegression(train_x_regressor, train_y_regressor, test_x_regressor):
    #     regressor_model = LogisticRegression(penalty='l1', dual=False, tol=0.0001, 
#                          C=1e20, fit_intercept=True, intercept_scaling=1.0, 
#                          class_weight= None, random_state=None)
    
#     regressor_model = RandomForestRegressor(n_estimators= 2000,
#                                min_samples_split=15, min_samples_leaf= 5)
#     
    regressor_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                               min_samples_split=30,min_samples_leaf= 5, loss='lad')
# # 
#     regressor_model = ExtraTreesRegressor(n_estimators= 1500,
#                                min_samples_split=15, min_samples_leaf= 5)
    
#     regressor_model = Lasso(alpha= 7)
     
   
    
#     train_x_regressor, train_y_regressor = com_stat.train_sample_boostrap(train_x_regressor, train_y_regressor)
#     train_x_regressor, train_y_regressor = adsys.generateSamples(train_x_regressor, train_y_regressor)
#     
    sample_weight = com_stat.stat_sample_weight(train_y_regressor)
    
    try:
        regressor_model.fit(train_x_regressor, train_y_regressor,sample_weight)
    except :
        print 'sample weight parameter is not legal...'
        regressor_model.fit(train_x_regressor, train_y_regressor)
        
#     try:     
#         print com_stat.feature_importance_stat(regressor_model.feature_importances_)
#     except:
#         print 'have not feature_importance_'

    
    # stage1: first regressor for all samples;   
    test_y_regressor_preds = regressor_model.predict(test_x_regressor)
    
    return test_y_regressor_preds


def get_model_cv_preds(model, X_train, y_train):
           
        kfold = cross_validation.StratifiedKFold(y_train, 5)
        stack_preds = []
        indexes_cv = []
        print 'cv stage preds...'
        for stage0, stack in kfold:
                    
            sample_weight = com_stat.stat_sample_weight(y_train[stage0])
            try:
                model.fit(X_train[stage0], y_train[stage0],sample_weight)
                
            except :
                print 'sample weight parameter is not legal...'
                model.fit(X_train[stage0], y_train[stage0])
            
            stack_preds.extend(list(model.predict_proba(X_train[stack])[:,1]))
            indexes_cv.extend(list(stack))
        print 'end stage..'    
        stack_preds = np.array(stack_preds)[sp.argsort(indexes_cv)]           

        return stack_preds
       
def classify_predictor(train_x, train_y, test_x):
    
    train_y_classify = np.asarray([1 if item > 0 else 0 for item in train_y])
    
    
    clf_model = LogisticRegression(penalty='l1', dual=False, tol=0.0001, 
                         C=1e20, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight='auto', random_state=None)
#     
    clf_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.3,
                               min_samples_split=30, min_samples_leaf= 5)
    
#     clf_model = ExtraTreesClassifier(n_estimators= 50, max_features = 5,
#                                min_samples_split=300, min_samples_leaf= 50,  
#                                max_depth = 20)
#     
#     clf_model = RandomForestClassifier(n_estimators= 50, max_features = 5,
#                                min_samples_split=20, min_samples_leaf= 5,  
#                                max_depth = 20)
#    
    train_x_clf = feature_generator.get_clf_features(train_x, is_test = 0)
    
    # oversampling for unbalance learning;
#     train_x_clf, train_y_classify = adsys.generateSamples(train_x_clf, train_y_classify)
    
    sample_weight = com_stat.stat_sample_weight(train_y_classify)
    try:
        clf_model.fit(train_x_clf, train_y_classify,sample_weight)
    except :
        print 'clf sample weight parameter is not legal...'
        clf_model.fit(train_x_clf, train_y_classify)
        
    print 'classcify feature size:',train_x_clf.shape
#     print clf_model.feature_importances_
    
    test_x_clf = feature_generator.get_clf_features(test_x, is_test = 1)
   
    test_y_preds_proba = clf_model.predict_proba(test_x_clf)[:,1]    
    
    test_y_preds = clf_model.predict(test_x_clf)
    
    # i think here must user cv get train sample preds
#     train_y_preds_proba = get_model_cv_preds(clf_model, train_x_clf, train_y_classify)
    train_y_preds_proba = None
    
    
    return test_y_preds, test_y_preds_proba


def get_train_regression_sets(train_x, train_y):
      
    train_regressor_index = np.where(train_y >0)[0]
    
    train_x_regressor = train_x[train_regressor_index]
    train_y_regressor = train_y[train_regressor_index] 
    
    # add little y= 0 sampls;
    y_zero_index = np.where(train_y == 0)[0]
#     x_new = []
#     y_new = []
#     for i in range(100):
#         ind = np.random.choice(y_zero_index)        
#         x_new.append(train_x[ind,:])
#         y_new.append(train_y[ind])
#     
#     
#     train_x_regressor = np.vstack((train_x_regressor, np.asarray(x_new)))
#     train_y_regressor = np.hstack((train_y_regressor, y_new))
    
    
    
    zeor_sample_indx = random.sample(y_zero_index, 300)
    train_x_regressor = np.vstack((train_x_regressor, train_x[zeor_sample_indx,]))
    train_y_regressor = np.hstack((train_y_regressor,  train_y[zeor_sample_indx]))
    
    return train_x_regressor, train_y_regressor



def classify_and_regressor_predictor(train_x, train_y, test_x, threshold = 5):    
    
    # classification model ;     
    test_y_preds, test_y_preds_proba = classify_predictor(train_x, train_y, test_x)    
    
    # regression model;
    train_x_regressor, train_y_regressor = get_train_regression_sets(train_x, train_y)    
    
            
    train_x_regressor = feature_generator.get_regressor_features(train_x_regressor,   is_test = 0)
    
    test_regressor_index = np.where(test_y_preds ==  1)[0]    
    test_x_regressor = test_x[test_regressor_index]    
    
    
    print 'test size:',test_x_regressor.shape
    test_x_regressor = feature_generator.get_regressor_features(test_x_regressor, is_test = 1)
    
    train_regressor_non_tail_index = np.where(train_y_regressor <= 100)[0]
    train_x_regressor_non_tail = train_x_regressor[train_regressor_non_tail_index]
    train_y_regressor_non_tail = train_y_regressor[train_regressor_non_tail_index]
    
#     test_y_regressor_preds = Quantile_Regression(train_x_regressor, train_y_regressor , test_x_regressor)
#     test_y_regressor_preds = NonlinearRegression(train_x_regressor_non_tail, np.log2(train_y_regressor_non_tail + 1), test_x_regressor)
# 
    test_y_regressor_preds = NonlinearRegression_CV(train_x_regressor_non_tail, np.log2(train_y_regressor_non_tail + 1.0), 
                                                     test_x_regressor,model_type ='GBR')
    
#     test_y_regressor_preds2 = NonlinearRegression_CV(train_x_regressor_non_tail, train_y_regressor_non_tail, 
#                                                      test_x_regressor,model_type ='ETR')
    
#     test_y_regressor_preds = test_y_regressor_preds1 

#     test_y_regressor_preds = ensemble_regression_preditor(train_x_regressor_non_tail, train_y_regressor_non_tail, test_x_regressor)
        
    # stage 2: re-regressor long-tail sample;
#     test_y_regressor_preds = long_tail_regressor_predictor(train_x_regressor, train_y_regressor, test_x_regressor, test_y_regressor_preds)
    
    test_y_regressor_preds = predict.postprocess_predict(np.power(2,test_y_regressor_preds) - 1.0, threshold = threshold)
    
    preds = np.asarray([0.0] * test_x.shape[0])
    
    
    
    preds[test_regressor_index] = test_y_regressor_preds
    
    return test_y_preds_proba, preds
        
  
def classify_and_regressor_model(is_test = 0 ):
    print 'classify and regressor model ...'
        
    train_data = data_io.read_train()
    
    feature_names = list(train_data.columns)
    feature_names.remove('id')   
    feature_names.remove('loss')
    
    
    train_x = train_data[feature_names].values
    
    train_y = train_data['loss'].values
    
    threshold = 1.0
    
    if not is_test:
        cv.classify_and_regressor_cv(classify_and_regressor_predictor, train_x, train_y, fig = 1, K =5, threshold = threshold)
#         cv.classify_and_regressor_cv(ensemble_preditor, train_x, train_y, fig = 0, K =5, threshold = threshold)
    else:
        test_data = data_io.read_test()
        
        feature_names = list(test_data.columns)
        feature_names.remove('id')        
        test_x = test_data[feature_names].values
        
        clf_preds_proba, preds = classify_and_regressor_predictor(train_x, train_y, test_x, threshold = threshold)
#         clf_preds_proba, preds = ensemble_preditor(train_x, train_y, test_x, threshold = threshold)
        
        test_ids = test_data['id'].values
      
        data_io.write_submission(test_ids, preds)
        com_stat.target_hist_dis(preds)
        
        
    

if __name__ == '__main__':
    
    classify_and_regressor_model(is_test = 0 )
    
    

    
    
    