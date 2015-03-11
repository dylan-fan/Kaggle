'''
Created on 2014/02/28

@author: dylanfan

'''

import scipy as sp
import numpy as np

from sklearn import cross_validation, linear_model

from feature_generator import get_dataset
import com_stat

class Stacked_Generalization(object):
    
    """
    Implement stacking to combine several models.
    The base (stage 0) models can be either combined through
    simple averaging (fastest), or combined using a stage 1 generalizer
    (requires computing CV predictions on the train set).

    See http://ijcai.org/Past%20Proceedings/IJCAI-97-VOL2/PDF/011.pdf:
    "Stacked generalization: when does it work?", Ting and Witten, 1997

    For speed and convenience, both fitting and prediction are done
    in the same method fit_predict; this is done in order to enable
    one to compute metrics on the predictions after training each model without
    having to wait for all the models to be trained.

    Options:
    ------------------------------
    - models: a list of (model, dataset) tuples that represent stage 0 models
    - generalizer: an Estimator object. Must implement fit and predict
  
    """
    
    def __init__(self, models, generalizer=None,
                 stack=False):
        
        self.models = models
      
        self.stack = stack
       
#         self.generalizer = linear_model.RidgeCV(
#             alphas=np.linspace(0, 200), cv=100)
        
        self.generalizer = linear_model.LogisticRegression(fit_intercept = False)
        
        
        

    def _combine_preds(self, X_train, X_cv, y,
                       stack=False):
        
        mean_preds = np.mean(X_cv, axis=1)
        stack_preds = None
     

        if stack:
            self.generalizer.fit(X_train, y)
            stack_preds = self.generalizer.predict(X_cv)
            
        return mean_preds, stack_preds

    
    def _get_model_preds(self, model, X_train, X_predict, y_train, is_cross_preds = 1):
       
        """
        Return the model predictions on the prediction set,
       
        """
        if not is_cross_preds:
            sample_weight = com_stat.stat_sample_weight(y_train)
            
            try:
                model.fit(X_train, y_train,sample_weight)
                
            except :
                print 'sample weight parameter is not legal...'
                model.fit(X_train, y_train)
                
            model_preds = model.predict(X_predict)
            
        else:
            kfold = cross_validation.StratifiedKFold(y_train, 10)
            stack_preds = []
           
          
            for stage0, stack in kfold:
                        
                sample_weight = com_stat.stat_sample_weight(y_train[stage0])
                try:
                    model.fit(X_train[stage0], y_train[stage0],sample_weight)
                    
                except :
                    print 'sample weight parameter is not legal...'
                    model.fit(X_train[stage0], y_train[stage0])
                    
                test_y_regressor_preds = model.predict(X_predict)
                stack_preds.append(test_y_regressor_preds)
    
            model_preds = np.median(np.array(stack_preds).T,axis=1)
            
            
        return model_preds
    

    def _get_model_cv_preds(self, model, X_train, y_train):
      
      
        kfold = cross_validation.StratifiedKFold(y_train, 4)
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
            
            stack_preds.extend(list(model.predict(
                X_train[stack])))
            indexes_cv.extend(list(stack))
        print 'end stage..'    
        stack_preds = np.array(stack_preds)[sp.argsort(indexes_cv)]           

        return stack_preds
    

    def fit_predict(self, y, train=None, predict=None):
       
        y_train = y[train] if train is not None else y
        if train is not None and predict is None:
            predict = [i for i in range(len(y)) if i not in train]

        stage0_train = []
        stage0_predict = []
        for model, feature_set in self.models:
            X_train, X_predict = get_dataset(feature_set, train, predict)
           
            model_preds = self._get_model_preds(
                model, X_train, X_predict, y_train)
            stage0_predict.append(model_preds)
            
        

            # if stacking, compute cross-validated predictions on the train set
            if self.stack:
                model_cv_preds = self._get_model_cv_preds(
                    model, X_train, y_train)
                stage0_train.append(model_cv_preds)
                

          

        mean_preds, stack_preds = self._combine_preds(
            np.array(stage0_train).T, np.array(stage0_predict).T,
            y_train, stack=self.stack)

        if self.stack:
            selected_preds = stack_preds 
        else:
            selected_preds = mean_preds

        return selected_preds

