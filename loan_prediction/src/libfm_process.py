'''
Created on 2013/09/22

@author: dylan
'''
import string
import os
from sklearn.datasets  import dump_svmlight_file
import data_io

import numpy as np

def read_preds_file(preds_f):
    fin = open(preds_f, 'r')
    predict_list = []
    for line in fin:
        pred = line.strip('\n')
        pred = string.atof(pred)
        predict_list.append(pred)
    fin.close()
    return np.asarray(predict_list)
    


def get_predict_probs(X_train,y_train, X_test,y_test, train_file, test_file, preds_file):
    dump_svmlight_file(X_train, y_train, f = train_file, zero_based=False)
    dump_svmlight_file(X_test, y_test, f = test_file, zero_based=False)
    
    # test is so large
    line_theta = 50000
    f_test = open(test_file,'r')
    tempfile = data_io.get_paths()['temp_test']
    o_f = open(tempfile,'w')
    
    i = 0
    predictions = []
    
    for line in f_test:
        if i < line_theta:
            o_f.write(line)
            i += 1
        else:
            o_f.write(line)
            o_f.close()
            cmd = "libfm -task r -iter 100 -dim '1,1,50'  -train " + train_file + " -test " + tempfile + " -out " + preds_file 
            os.system(cmd)
            predictions.extend(read_preds_file(preds_file))
            i = 0
            o_f = open(tempfile,'w')
            
    
    if i != 0:
        o_f.close()
        cmd = "libfm -task r -iter 100 -dim '1,1,50'  -train " + train_file + " -test " + tempfile + " -out " + preds_file 
        os.system(cmd)
        predictions.extend(read_preds_file(preds_file))
    
    predictions = list(-1.0*np.asarray(predictions))
    
    
    return predictions






