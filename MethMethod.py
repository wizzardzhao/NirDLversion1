# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:19:58 2017

@author: liang
"""

def CalculateMixtureMetrics(test_y_s, predict,averageMode):
    from sklearn import metrics 
    precision = metrics.precision_score(test_y_s, predict,average = averageMode)    
    recall = metrics.recall_score(test_y_s, predict,average = averageMode)    
    print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))    
    accuracy = metrics.accuracy_score(test_y_s, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))

    return precision,recall,accuracy