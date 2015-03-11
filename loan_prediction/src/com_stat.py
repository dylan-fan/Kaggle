'''
Created on 2014/01/29

@author: dylan
'''


import numpy as np
import pylab as pl
import data_io


def roc_curve(fpr, tpr):
    pl.figure()
  
    pl.plot(fpr, tpr)
    pl.xlabel('fpr')
    pl.ylabel('tpr')
    pl.title('ROC')
    pl.show()
    pl.close()
    
def classify_predictor_prob_dis(preds_prob):
    pl.figure()  
    pl.hist(preds_prob,bins =50, log = True, normed = True)
    pl.xlabel('preds prob')    
    pl.title('preds prob hist')
    
    pl.show()
    pl.close()
    
    
    
def feature_importance_stat(feature_importance_li):
    feature_importane_sort_li = []
    for i, item in enumerate(feature_importance_li):
        feature_importane_sort_li.append((i, item))
    feature_importane_sort_li.sort(cmp=None, key=lambda x:x[1], reverse=True)
#     feature_ids = [fid for fid, imp in feature_importane_sort_li if imp > 0.0001]
# #     feature_ids.sort()
#     data_io.save_importanct_features(feature_ids)
    return feature_importane_sort_li

def stat_sample_weight(y):
#     target_hist_dis(y)
    
    sample_weight = []
    loss_dis_dict = get_loss_freqs_dict(y)    
    counts = np.sum(loss_dis_dict.values())
    
    for e in y:
        w = loss_dis_dict.get(e,2)
        
        if w < 5 or w/(counts + 1.0) < 0.5e-4:
            w = w * 2
       
        
        
#         if e > 40 and e < 60:
#             w = w * 3
#         if e> 60 and e < 90 :
#             w  = w * 2
#       
        w = w/(counts+ 1.0)      
        sample_weight.append(w)
    
    return np.asarray(sample_weight)



def get_loss_freqs_dict(y):
    loss_dis_dict = {}
   
    for e in y:
        loss_dis_dict.setdefault(e,0)
        loss_dis_dict[e] += 1
    
    return loss_dis_dict


def train_sample_boostrap(train_x ,train_y, is_add_noise = 0):
    loss_dis_dict = get_loss_freqs_dict(train_y)
    counts = np.sum(loss_dis_dict.values())
    train_y_new = [] 
    train_y_new.extend(train_y)
    
    print 'boostrap raw sample:', train_x.shape
    for i in range(len(train_y)):
        loss_count = loss_dis_dict.get(train_y[i])        
        rep_times = 0
       
        if train_y[i] >= 1 and train_y[i] <=10 :
            rep_times = 3
        
        if train_y[i] > 10 and train_y[i] < 30:
            rep_times  = 1
            
        for j in range(rep_times):
            # add gauss noise;
            if is_add_noise:                
                noise_samp = np.asarray([np.random.normal() * 0.001 for k in range(len(train_x[i]))])
                sample = train_x[i] + noise_samp
            
            else:
                sample = train_x[i,:]
                
            train_x = np.vstack((train_x,sample))
            train_y_new.append(train_y[i])
                
    print 'boostrap sample:', train_x.shape
    return train_x , np.asarray(train_y_new)
    
        
    
    
def target_hist_dis(target_vals):
    loss_dis_dict = {}
    for e in target_vals:
        if e < 1e-9:
            continue
        loss_dis_dict.setdefault(e,0)
        loss_dis_dict[e] += 1
        
    counts = np.sum(loss_dis_dict.values())
    for e , val in loss_dis_dict.iteritems():
        print e, val, val/ (counts + 0.0)
        
    pl.figure()
    pl.hist(target_vals,bins =50, log = True, normed = True)
    pl.title('loss hist')
    pl.show()
    pl.close()
    
def train_test_target_dis(train_target_vals, test_target_vals):
    pl.figure()
    pl.subplot(1,2,1)
   
    pl.hist(train_target_vals,bins =50, log = True, normed = True)
    pl.title(' train target hist ')
    
    pl.subplot(1,2,2)
    pl.hist(test_target_vals,bins =50, log = True, normed = True)
    pl.title('test target hist')
    
    pl.show()
    pl.close()
    
def target_preds_and_true_dis(preds, y_trues, remove_zero = 0):
    
    if remove_zero:
        preds = [e for e in preds if e >0]
        y_trues = [e for e in y_trues if e >0]
    
    err_dis_dict ={}
    for i in range(len(y_trues)):
        err_dis_dict.setdefault(y_trues[i],[0,0])
        err_dis_dict[y_trues[i]][0] += np.fabs(y_trues[i] - preds[i])
        err_dis_dict[y_trues[i]][1] += 1
    
    for e, val in err_dis_dict.iteritems():
        err_dis_dict[e][0] = val[0]/ (val[1] +0.0)
    
    print err_dis_dict.keys()
    print [item[0] for item in err_dis_dict.values()]
    print [item[1] for item in err_dis_dict.values()]
    
    pl.figure()
    pl.subplot(2,2,1)
    
        
    pl.hist(preds,bins =50, log = True, normed = True)
    pl.title('preds hist')
    
    pl.subplot(2,2,2)
    pl.hist(y_trues,bins =50, log = True, normed = True)
    pl.title('true target hist')
    
    pl.subplot(2,2,3)
    pl.plot(err_dis_dict.keys(), [item[0] for item in err_dis_dict.values()],'r')
    
    pl.title('error dis')
    pl.xlabel('y_trues')
    
    pl.subplot(2,2,4)
    pl.plot(y_trues,preds,'r*')
    pl.title('y_true vs preds')
    pl.xlabel('y_true')
    pl.ylabel('preds')
    
    pl.show()
    pl.close()
    
def target_preds_and_true_threshold_stat(preds, y_trues, threshold = 4):
    preds_threshold_n = 0
    for e in preds:
        if e <= threshold and e >= 1:
            preds_threshold_n += 1
            
    y_trues_threshold_n = 0
    for e in y_trues:
        if e <= threshold and e >=1:
            y_trues_threshold_n += 1
    
    print 'preds threshold:%d, sample_size: %d ,ratio:%f'%(threshold, preds_threshold_n, preds_threshold_n/ (len(preds)+0.0))
    print 'y_true threshold:%d, sample_size: %d ,ratio:%f'%(threshold, y_trues_threshold_n, y_trues_threshold_n/ (len(preds)+0.0))
            

def fig_feature(X):
    pl.figure()
    pl.hist(X[:,639],bins = 50,  normed = True)
    pl.show()
    pl.close()
    
    
    
if __name__ == '__main__':
    
    train_data = data_io.read_train()    
    train_y = train_data['loss'].values
#     for e in train_y:
#         if e != 0:
#             print e
#     train_y = data_io.read_train_f()
    target_hist_dis(train_y)