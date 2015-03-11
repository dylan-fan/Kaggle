'''
Created on 2014/01/28
@author: dylan
'''

import csv
import pickle
import pandas as pd
import json
import os
import numpy as np


def read_train_f(train_file = '../data/train.csv'):
    in_f = open(train_file,'r' )
    reader = csv.reader( in_f)
    reader.next()
    
    n = 0 
    y_zero = 0
    
    loss_li = []
    
    feature_test_f = open('../data/feature_test.txt','w')
    for line in reader:
        n += 1
        if n == 1:
            for e in line:
                print >> feature_test_f, e
            
        if int(line[-1]) == 0:
            y_zero += 1
        
        loss_li.append(int(line[-1]))
            
            
    feature_test_f.close()
    
    print 'train samples are: ',n
    print 'train samples with y = 0 are: ', y_zero
    print 'train samples ratio with y = 0 are: ', y_zero/(n + 0.0)
    
    return loss_li
    
def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths    

def read_train():
    train_path = get_paths()["train_path"]
    
    return pd.read_csv(train_path)

def read_test():
    test_path = get_paths()["test_path"]
    return pd.read_csv(test_path)



def load_train_feature():
    feature_path = get_paths()['train_feature']
    return pd.read_csv(feature_path).values

def load_test_feature():
    feature_path = get_paths()['test_feature']
    return pd.read_csv(feature_path).values

     
def save_train_feature(X):
    feature_path = get_paths()['train_feature']
    data = pd.DataFrame(X)
    data.to_csv(feature_path, header = True, index = False)
    

def save_test_feature(X):
    feature_path = get_paths()['test_feature']
    data = pd.DataFrame(X)
    data.to_csv(feature_path, header = True, index = False)
    

def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, "w"))    


def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path))

def save_scale_model(model):
    out_path = get_paths()["scale_model_path"]
    pickle.dump(model, open(out_path, "w"))
    
def load_scale_model():
    in_path = get_paths()["scale_model_path"]
    return pickle.load(open(in_path))

def save_decomposition_model(model):
    out_path = get_paths()["decomposition_model_path"]
    pickle.dump(model, open(out_path, "w"))
    
def load_decomposition_model():
    in_path = get_paths()["decomposition_model_path"]
    return pickle.load(open(in_path))

def save_reg_importanct_features(feature_ids):
    out_path = get_paths()["reg_import_feature_ids"]
    pickle.dump(feature_ids, open(out_path, "w"))
    
def load_reg_importanct_features(): 
    in_path = get_paths()["reg_import_feature_ids"]
    return pickle.load(open(in_path))

def save_clf_importanct_features(feature_ids):
    out_path = get_paths()["clf_import_feature_ids"]
    pickle.dump(feature_ids, open(out_path, "w"))
    
def load_clf_importanct_features(): 
    in_path = get_paths()["clf_import_feature_ids"]
    return pickle.load(open(in_path))

def write_submission(ids, preds):
    csvfile = open(get_paths()['submission_path'],'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['id','loss'])
    
    for i in range(len(ids)):
        writer.writerow([ids[i], preds[i]])
    csvfile.close()
def read_quantile_regression_preds():
    in_path = get_paths()['quantile_regression_preds']
    preds = np.asarray(pd.read_csv(in_path).values, dtype = float)
    
    return preds.T[0]
    
        
    
if __name__ == '__main__':
    read_quantile_regression_preds()