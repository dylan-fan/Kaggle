'''
Created on 2014/01/28
@author: dylan

'''

import data_io
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from scipy import stats

import com_stat
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA




import logging
import cv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                    filename="history.log", filemode='a', level=logging.DEBUG,
                    datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                              datefmt='%m/%d/%y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logger = logging.getLogger(__name__)


def stats_importance_regressor_features():
    
    train_data = data_io.read_train()
    
    feature_names = list(train_data.columns)
    feature_names.remove('id')
    feature_names.remove('loss') 
    
    train_x = train_data[feature_names].values
    
    train_y = train_data['loss'].values
    train_regressor_index = np.where(train_y > 0)[0]
    
    train_x  = train_x[train_regressor_index]
    y = train_y[train_regressor_index]
    
    X = np.array([train_x[:,521] - train_x[:,520]]).T    
    X = np.hstack((X,np.array([train_x[:,521] + train_x[:,520]]).T))
    X = np.hstack((X,np.array([train_x[:,521] - train_x[:,271]]).T))
    X = np.hstack((X,np.array([train_x[:,521] + train_x[:,268]]).T))
    X = np.hstack((X,train_x[:,0:train_x.shape[1]]))
    
    
    
    print 'previous impute : train-x size:', X.shape
    
    X = np.asarray(X, dtype = float)
    col_mean = stats.nanmean(X,axis=0)
    inds = np.where(np.isnan(X))
    X[inds]=np.take(col_mean,inds[1])
    
    print 'post impute : train-x size:', X.shape
    
    scale_model = StandardScaler()
    X = scale_model.fit_transform(X) 
    
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.3,
                               min_samples_split=30,min_samples_leaf= 10,  loss='lad')  
    

    model.fit(X, y)       
   
    feature_importane_sort_li = com_stat.feature_importance_stat(model.feature_importances_)
    print feature_importane_sort_li
    feature_ids = [fid for fid, imp in feature_importane_sort_li if imp > 1e-5] 
    print 'import feature_ids size:', len(feature_ids)        
    data_io.save_reg_importanct_features(feature_ids)
    
    
    
    
def stats_importance_classify_features():
    
    train_data = data_io.read_train()
    
    feature_names = list(train_data.columns)
    feature_names.remove('id')
    feature_names.remove('loss')   
    
    sample_size = 10000
    
    train_x = train_data[feature_names].values
    
    train_y = train_data['loss'].values[0:sample_size]
    
    train_y_classify = [1 if item > 0 else 0 for item in train_y]
    
    train_x = train_x[0:sample_size,0:train_x.shape[1]]
   
    X = np.array([train_x[:,521] - train_x[:,520]]).T    
    X = np.hstack((X,np.array([train_x[:,521] + train_x[:,520]]).T))
    X = np.hstack((X,np.array([train_x[:,521] - train_x[:,271]]).T))
    X = np.hstack((X,np.array([train_x[:,521] + train_x[:,268]]).T))
    
    X = np.hstack((X,np.array([train_x[:,46] - train_x[:,36]]).T))
    X = np.hstack((X,train_x[:,0:train_x.shape[1]]))
    
    print 'previous impute : train-x size:', X.shape
    
    X = np.asarray(X, dtype = float)
    col_mean = stats.nanmean(X,axis=0)
    inds = np.where(np.isnan(X))
    X[inds]=np.take(col_mean,inds[1])
    
    print 'post impute : train-x size:', X.shape
    
    scale_model = StandardScaler()
    X = scale_model.fit_transform(X) 
    
    clf_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.3,
                               min_samples_split=30, min_samples_leaf= 5)   
    
    clf_model = LogisticRegression(penalty='l1', dual=False, tol=0.0001, 
                         C=1e20, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight='auto', random_state=None)
#     clf_model.fit(X, train_y_classify)        
   
#     feature_importane_sort_li = com_stat.feature_importance_stat(clf_model.feature_importances_)
#     print feature_importane_sort_li
#     feature_ids = [fid for fid, imp in feature_importane_sort_li if imp > 1e-5] 
#     print 'import feature_ids size:', len(feature_ids)
#         
#     data_io.save_clf_importanct_features(feature_ids)
#     X = X[:,feature_ids]

    score = cv.cv_loop(X, train_y_classify, clf_model, N= 5)
    print "Mean AUC: %f" % (score)
    
    
def greedy_classify_features():
    
    train_data = data_io.read_train()
    
    feature_names = list(train_data.columns)
    feature_names.remove('id')
    feature_names.remove('loss')   
    
    sample_size = 20000
    
    train_x = train_data[feature_names].values
    
    train_y = train_data['loss'].values[0:sample_size]
    
    train_y_classify = [1 if item > 0 else 0 for item in train_y]
    
    train_x = train_x[0:sample_size,0:train_x.shape[1]]
   
    X = np.array([train_x[:,521] - train_x[:,520]]).T    
    X = np.hstack((X,np.array([train_x[:,521] + train_x[:,520]]).T))
    X = np.hstack((X,np.array([train_x[:,521] - train_x[:,271]]).T))
    X = np.hstack((X,np.array([train_x[:,521] + train_x[:,268]]).T))
    X = np.hstack((X,train_x[:,0:train_x.shape[1]]))
    
    print 'previous impute : train-x size:', X.shape
    
    X = np.asarray(X, dtype = float)
    col_mean = stats.nanmean(X,axis=0)
    inds = np.where(np.isnan(X))
    X[inds]=np.take(col_mean,inds[1])
    
    print 'post impute : train-x size:', X.shape
    
    scale_model = StandardScaler()
    X = scale_model.fit_transform(X) 
   
    
    clf_model = LogisticRegression(penalty='l1', dual=False, tol=0.0001, 
                         C=1e20, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight='auto', random_state=None)

    good_features = set([])
    score_hist = []
    
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        scores = []
        for f in range(X.shape[1]):
            if f not in good_features:                
                feats = list(good_features)+[f]
                print feats
                Xt = X[:,feats]
                print Xt.shape
                score = cv.cv_loop(Xt, train_y_classify, clf_model, N= 5)
                scores.append((score, f))
                print "Feature: %i Mean AUC: %f" % (f, score)
        scores.sort(cmp=None, key=lambda x : x[0], reverse=True)
        good_features.add(scores[0][1])
        score_hist.append(scores[0])
        print 'score_hist:',score_hist
        print "Current features: %s" % sorted(list(good_features))
        
    data_io.save_clf_importanct_features(list(good_features))
    

def get_features(datas, is_test = 0):
    feature_names = list(datas.columns)
    feature_names.remove('id')
    print 'f271:', feature_names.index('f271')
    print 'f274:', feature_names.index('f274')
    print 'f527:', feature_names.index('f527')
    
    print 'f49:', feature_names.index('f49')
    print 'f39:', feature_names.index('f39')
    
    feature_names= ['f527','f528']
    X = np.asarray(datas[feature_names].values, dtype = float)
    imp = Imputer(strategy='median', axis=0)
    
#     X = np.array([X[:,1] - X[:,0]]).T
    X = np.hstack((X,np.array([X[:,1] - X[:,0]]).T))
    X = imp.fit_transform(X)
    
    if not is_test:
        scale_model = StandardScaler()
        X = scale_model.fit_transform(X) 
        data_io.save_scale_model(scale_model)
    
    else:
        scale_model = data_io.load_scale_model()
        X = scale_model.transform(X) 
        
    
    
    return X

def get_clf_features(datas, is_test = 0):    
    
    X = np.array([datas[:,521] - datas[:,520]]).T    
    X = np.hstack((X,np.array([datas[:,521] + datas[:,520]]).T))
    X = np.hstack((X,np.array([datas[:,521] - datas[:,271]]).T))
    X = np.hstack((X,np.array([datas[:,521] + datas[:,268]]).T))
    X = np.hstack((X,datas[:,0:datas.shape[1]]))
    
    
    print 'previous impute : train-x size:', X.shape
 
     
    X = np.asarray(X, dtype = float)
    col_mean = stats.nanmean(X,axis=0)
    inds = np.where(np.isnan(X))
    X[inds]=np.take(col_mean,inds[1])
     
    print 'post impute : train-x size:', X.shape
    
#     X = features_preprocess(X)
    
    
    import_feature_ids = data_io.load_clf_importanct_features()  
    
    top_n = len(import_feature_ids)  
    top_n = 2
    X = X[:,import_feature_ids[:top_n]]   
    
    if not is_test:
        scale_model = StandardScaler()
        X = scale_model.fit_transform(X) 
        data_io.save_scale_model(scale_model)
    
    else:
        scale_model = data_io.load_scale_model()
        X = scale_model.transform(X) 
   
    
    
    return X
    

    
    
def get_regressor_features(datas, clf_proba = None, is_test = 0, is_decomp = 0):
        
    X = np.array([datas[:,521] - datas[:,520]]).T    
    X = np.hstack((X,np.array([datas[:,521] + datas[:,520]]).T))
    X = np.hstack((X,np.array([datas[:,521] - datas[:,271]]).T))
    X = np.hstack((X,np.array([datas[:,521] + datas[:,268]]).T))
    X = np.hstack((X,datas[:,0:datas.shape[1]]))  
    
    if clf_proba != None:
        X = np.hstack((X, np.asarray([clf_proba]).T))  
        
#     print 'imputer previous , feature size: ', X.shape[1]
#     
#     X = np.asarray(X, dtype = float)
#     col_mean = stats.nanmean(X,axis=0)
#     inds = np.where(np.isnan(X))
#     X[inds]=np.take(col_mean,inds[1])
#     
#     print 'imputer poster , feature size: ', X.shape[1]
    
    X = features_preprocess(X)
    
    
    import_feature_ids = data_io.load_reg_importanct_features()    
    
    top_n = len(import_feature_ids)  
    top_n = 150
    if clf_proba != None: 
        X = X[:,import_feature_ids[:top_n]+[X.shape[1] -1]]
    
    else:        
        X = X[:,import_feature_ids[:top_n]]
        
  
         
        
    if not is_test:
        scale_model = StandardScaler()
        X = scale_model.fit_transform(X) 
        data_io.save_scale_model(scale_model)
       
        if is_decomp:
#             decomposition_model = TruncatedSVD(n_components=8)
            decomposition_model = PCA(n_components=150)
            X = decomposition_model.fit_transform(X)
            data_io.save_decomposition_model(decomposition_model)
        
        
    
    else:
        scale_model = data_io.load_scale_model()
        X = scale_model.transform(X) 
        if is_decomp:
            decomposition_model = data_io.load_decomposition_model()
            X = decomposition_model.transform(X)
        
    
       
    
    print 'feature size: ',X.shape  
    
    return X
    

def features_preprocess(X):
    print 'imputer previous , feature size: ', X.shape[1]
    
    X = np.asarray(X, dtype = float)
    col_mean = stats.nanmean(X,axis=0)
    inds = np.where(np.isnan(X))
    X[inds]=np.take(col_mean,inds[1])
    
    print 'imputer poster , feature size: ', X.shape[1]
    
    for i in range(X.shape[1]):
        f_vals = X[:,i]
        val_percentile_min = stats.scoreatpercentile(f_vals,1)
        val_percentile_max = stats.scoreatpercentile(f_vals,99)
        for j in range(X.shape[0]):
            if X[j,i] < val_percentile_min:
                X[j,i] = val_percentile_min
            if X[j,i] > val_percentile_max:
                X[j,i] = val_percentile_max
        
    
    return X
            
    
    
    
def get_features_1(datas, is_test = 0):
    
    feature_names = list(datas.columns)
    feature_names.remove('id')
      
    print feature_names
    
    y_zero = 0
    n = 0
    if not is_test:
        feature_names.remove('loss')
        y_zero = np.sum([1 if item == 0 else 0 for item in datas['loss'].values])           
        
        n = len(datas)
        
        
        print 'train samples are: ',n
        print 'train samples with y = 0 are: ', y_zero
        print 'train samples ratio with y = 0 are: ', y_zero/(n + 0.0)
        
        train_y = datas['loss'].values
        

    str_feature_li = ['f11', 'f12',
                       'f462', 'f463', 'f473', 'f474', 'f602', 'f603', 'f605']
    
    print datas['f419'].values
  
    
    for f_name in feature_names:        
        test_f = datas[f_name].values
        i = 0
        for e in test_f:
            i += 1             
            if isinstance(e, str):
                print f_name, i, e
                str_feature_li.append(f_name)
                break
     
    for f_name in str_feature_li:
        try:
            feature_names.remove(f_name)
        except:
            continue
        
        
    
    unrelated_feature_li = []
    
    for f_name in feature_names:
        f_vals = datas[f_name].values
        f_vals = f_vals[~np.isnan(f_vals)]
        
        f_min_val = np.min(f_vals)
        f_max_val = np.max(f_vals)
       
        if f_min_val == f_max_val:
            unrelated_feature_li.append(f_name)
            
    
    for f_name in unrelated_feature_li:          
        feature_names.remove(f_name)
        
#     str_feature_li = [] 
#     
#     for f_name in feature_names:
#         f_vals = datas[f_name].values
#         f_vals = f_vals[~np.isnan(f_vals)]
#         i = 0
#         for e in f_vals:
#             i += 1             
#             if isinstance(e, float) :
#                 break
#         
#         if i < len(f_vals):
#             continue
#         
#         else:
#             f_min_val = np.min(f_vals)
#             f_max_val = np.max(f_vals)
#             f_median_val = np.median(f_vals)
#             print f_name, f_min_val,f_median_val,f_max_val
#                         
#             if f_max_val > 1e9:                
#                 str_feature_li.append(f_name)
                  
#     for f_name in str_feature_li:
#         feature_names.remove(f_name)
     
    print 'features size: ', len(feature_names)  
    
    
    for f_name in feature_names:
        f_vals = datas[f_name].values
        
        
        mean_val = np.mean(f_vals[~np.isnan(f_vals)])
        min_val = np.min(f_vals[~np.isnan(f_vals)])
        median_val = np.median(f_vals[~np.isnan(f_vals)])
        max_val = np.max(f_vals[~np.isnan(f_vals)])
        
        val_5 = stats.scoreatpercentile(f_vals[~np.isnan(f_vals)], 5)
        
        default_val =  val_5
        
        if max_val > 1e7 :
            print 'f: max, min, mean, median: ',f_name, max_val,min_val,mean_val, median_val
            print 'feature vals set len : ',len(set(f_vals[~np.isnan(f_vals)])) 
        
        
        datas[f_name].fillna(default_val,inplace = True)
    
   
    flag = 0
    for f_name in feature_names:
        f_vals = datas[f_name].values
        if np.max(f_vals) > 1e5 and np.min(f_vals) >= 1e-6:
            f_vals = np.log2(f_vals + 1.0)

        val_percentile_min = stats.scoreatpercentile(f_vals,1)
        val_percentile_max = stats.scoreatpercentile(f_vals,90)
       
        
        f_vals_adj = []
        
        for e in f_vals:
            adj_val = e             
            if e < val_percentile_min:
                adj_val = val_percentile_min
            if e > val_percentile_max:
                adj_val = val_percentile_max
            
            f_vals_adj.append([adj_val])
                
        if flag == 0:
            X = f_vals_adj
            flag = 1    
        
        else:
            X = np.hstack((X,f_vals_adj))
        
   
    print X
    scale_model = StandardScaler()
    X = scale_model.fit_transform(X)  
    
    return X


def create_datasets(train_x_regressor_feature, test_x_regressor_feature):    
    
    data_io.save_train_feature(train_x_regressor_feature)
    data_io.save_test_feature(test_x_regressor_feature)



def get_dataset(feature_set, train, cv):
    """
    Return the design matrices constructed with the specified feature set.
    If train is specified, split the training set according to train and
    cv (if cv is not given, subsample's complement will be used instead).
    If subsample is omitted, return both the full training and test sets.
    """
    try:       
        if train is not None:
            X = data_io.load_train_feature()
            if cv is None:
                cv = [i for i in range(X.shape[0]) if i not in train]

            X_test = X[cv, :]
            X = X[train, :]
        else:
            X = data_io.load_train_feature()
            X_test = data_io.load_test_feature()
            
    except IOError:
        logging.warning("could not find feature set %s", feature_set)
        return False

    return X, X_test
 
     
if __name__ == '__main__':    
    
#     greedy_classify_features()
    datas = data_io.read_train()
    get_features(datas, is_test = 0) 
    stats_importance_classify_features()
#     stats_importance_regressor_features()
    