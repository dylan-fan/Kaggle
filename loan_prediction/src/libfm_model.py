'''
Created on 2014/02/13

@author: dylanfan

'''
import data_io
from sklearn.datasets  import dump_svmlight_file
import feature_generator
import libfm_process
import cv
import numpy as np
import os
import libfm_process
import predict

import csv

def convert_to_libfm_format(features,target, is_test = 0):
    if not is_test:
        dump_svmlight_file(features, target, f = data_io.get_paths()['svmlight_format_train'], zero_based = False)
    else:
        dump_svmlight_file(features, target, f = data_io.get_paths()['svmlight_format_test'], zero_based = False)


def libfm_model(is_test = 0):
    train_data = data_io.read_train()
    train_x = feature_generator.get_features(train_data, is_test = 0)
    train_y = train_data['loss'].values
    
    
    threshold = 10
    if not is_test:
#         do cross validation;
        cv.libfm_cv(train_x, train_y, fig = 1, K =5 , threshold = threshold)
    
    else:
        test_data = data_io.read_test()
        test_x = feature_generator.get_features(test_data, is_test = 1)
        test_y = np.zeros((test_x.shape[0],1))
        
        
        train_file = data_io.get_paths()['svmlight_format_train']
        test_file = data_io.get_paths()['svmlight_format_test']
        preds_file = data_io.get_paths()['libfm_preds']
        
        preds = libfm_process.get_predict_probs(train_x, train_y, test_x, test_y, train_file, test_file, preds_file)        
       
        preds = predict.postprocess_predict(preds, threshold = threshold)
        
        
        test_ids = test_data['id'].values
        data_io.write_submission(test_ids, preds)
        
 
if __name__ == '__main__':
    libfm_model( is_test = 0)
    
    