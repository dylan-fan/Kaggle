'''
Created on 2014/02/11

@author: dylanfan
'''

import math

def __gaussion(dist, sigma=40.0):
    return math.e ** (-dist / (2 * sigma ** 2 + 0.0))

def get_gaussion_weight_li(distance_li):
    weight_li = []
    for e in distance_li:
        gauss_weight = __gaussion(e, sigma = 3)
        weight_li.append(gauss_weight)
#         print e
#         print gauss_weight
    
    return weight_li


if __name__ == '__main__':
    pass