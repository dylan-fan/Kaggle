'''
Created on 2014/03/09

@author: dylan
'''
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random



def get_class_count(y, minorityclasslabel = 1):
    minorityclasslabel_count = len(np.where(y == minorityclasslabel)[0])
    maxclasslabel_count = len(np.where(y == (1 - minorityclasslabel))[0])
    
    return maxclasslabel_count, minorityclasslabel_count
    
    
# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @param ms: The amount of samples in the minority group
# @param ml: The amount of samples in the majority group
# @return: the G value, which indicates how many samples should be generated in total, this can be tuned with beta
def getG(ml, ms, beta):
    return (ml-ms)*beta


# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @param: minorityclass: The minority class
# @param: K: The amount of neighbours for Knn
# @return: rlist: List of r values
def getRis(X,y,indicesMinority,minorityclasslabel,K):    
    
    ymin = np.array(y)[indicesMinority]
    Xmin = np.array(X)[indicesMinority]
    neigh = NearestNeighbors(n_neighbors= K)
    neigh.fit(X)
    
    rlist = [0]*len(ymin)
    normalizedrlist = [0]*len(ymin)
    
    for i in xrange(len(ymin)):
        indices = neigh.kneighbors(Xmin[i],K,False)[0]
#         print'y[indices] == (1 - minorityclasslabel):'
#         print y[indices]
#         print len(np.where(y[indices] == ( 1- minorityclasslabel))[0])
        rlist[i] = len(np.where(y[indices] == ( 1- minorityclasslabel))[0])/(K + 0.0)
        
    normConst = sum(rlist)

    for j in xrange(len(rlist)):
        normalizedrlist[j] = (rlist[j]/normConst)

    return normalizedrlist

def get_indicesMinority(y, minorityclasslabel = 1):
    y_new = []
    for i in range(len(y)):
#         if y[i] > 20 and y[i] < 80:
        if y[i] == 1:
            y_new.append(1)
        else:
            y_new.append(0)
    y_new = np.asarray(y_new)
    indicesMinority = np.where(y_new == minorityclasslabel)[0] 
       
    return indicesMinority, y_new

def generateSamples(X, y, minorityclasslabel = 1, K =5):
    syntheticdata_X = []
    syntheticdata_y = []
    
    
    indicesMinority, y_new = get_indicesMinority(y)
    ymin = y[indicesMinority]
    Xmin = X[indicesMinority]
    
    rlist = getRis(X, y_new, indicesMinority, minorityclasslabel, K)
    ml, ms = get_class_count(y_new)
    G = getG(ml,ms, beta = 0.5)
   
    neigh = NearestNeighbors(n_neighbors=K)
    neigh.fit(Xmin)
    
    for k in xrange(len(ymin)):
        g = int(np.round(rlist[k]*G))

        neighb_indx = neigh.kneighbors(Xmin[k],K,False)[0]
            
        for l in xrange(g):
            ind = random.choice(neighb_indx)
            s = Xmin[k] + (Xmin[ind]-Xmin[k]) * random.random()
            syntheticdata_X.append(s)
            syntheticdata_y.append(ymin[k])
            
    print 'asyn, raw X size:',X.shape        
    X = np.vstack((X,np.asarray(syntheticdata_X)))
   
    y = np.hstack((y,syntheticdata_y))
    print 'asyn, post X size:',X.shape
    
    return X , y
   

if __name__ == '__main__':
    pass