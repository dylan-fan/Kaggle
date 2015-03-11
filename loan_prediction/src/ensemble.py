'''
Created on 2014/02/25

@author: dylanfan

'''


import scipy as sp
import numpy as np
import feature_generator
import data_io

from sklearn import cross_validation, linear_model
from Stacked_Generalization import Stacked_Generalization

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor

import predict

def ensemble_regression_preditor(train_x_regressor, train_y_regressor,test_x_regressor, stack = False):
    
    selected_models = [
        "GBR:tuples_sf",
        "ETR:fe"                
    ]
    
    models = []
    model_dict = {'GBR': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                               min_samples_split=30,min_samples_leaf= 5, loss='lad'),
                  
                  'ETR': ExtraTreesRegressor(n_estimators= 1500,
                               min_samples_split=15, min_samples_leaf= 5)               
                 }
    
    for item in selected_models:
        model_id, dataset = item.split(':')
        model = model_dict[model_id]
        models.append((model,dataset))
    
    feature_generator.create_datasets(train_x_regressor, test_x_regressor)
    
    
    ensemble_regression = Stacked_Generalization(models = models, stack = stack)
    test_y_regressor_preds = ensemble_regression.fit_predict(train_y_regressor)   
    
    
    return test_y_regressor_preds

    
def ensemble_preditor(train_x, train_y, test_x, threshold = 5, stack = False):
    
    train_y_classify = [1 if item > 0 else 0 for item in train_y]
    
    
  
    clf_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.3,
                               min_samples_split=30, min_samples_leaf= 5)
    

    
    train_x_clf = feature_generator.get_clf_features(train_x, is_test = 0)

    
    clf_model.fit(train_x_clf, train_y_classify)
    
    test_x_clf = feature_generator.get_clf_features(test_x, is_test = 1)

   
    test_y_preds_proba = clf_model.predict_proba(test_x_clf)[:,1]
    
    
    train_regressor_index = np.where(train_y > 0)[0]
    
    train_x_regressor = train_x[train_regressor_index]
    train_y_regressor = train_y[train_regressor_index]
    
    train_x_regressor = feature_generator.get_regressor_features(train_x_regressor, is_test = 0)
    
    test_y_preds = clf_model.predict(test_x_clf)
    
    test_regressor_index = np.where(test_y_preds ==  1)[0]
    

    test_x_regressor = test_x[test_regressor_index]
    test_x_regressor = feature_generator.get_regressor_features(test_x_regressor, is_test = 1)
    
    
    selected_models = [
        "GBR:tuples_sf",
        "ETR:fe",
        "RFR:fe"
                
    ]
    
    models = []
    model_dict = {'GBR': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                               min_samples_split=30,min_samples_leaf= 5, loss='lad'),
                  
                  'ETR': ExtraTreesRegressor(n_estimators= 1500,
                               min_samples_split=15, min_samples_leaf= 5)               
                 }
    
    for item in selected_models:
        model_id, dataset = item.split(':')
        model = model_dict[model_id]
        models.append((model,dataset))
    
    feature_generator.create_datasets(train_x_regressor, test_x_regressor)
    
    
    ensemble_regression = Stacked_Generalization(models = models, stack = stack)
    test_y_regressor_preds = ensemble_regression.fit_predict(train_y_regressor)
    
    test_y_regressor_preds = predict.postprocess_predict(test_y_regressor_preds, threshold = threshold)
    preds = np.asarray([0.0] * test_x.shape[0])
    
    preds[test_regressor_index] = test_y_regressor_preds
    
    
    return test_y_preds_proba, preds
    

