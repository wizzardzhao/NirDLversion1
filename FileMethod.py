# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:22:00 2017

@author: liang
"""
def staticLabelNum(y_array):
    a = set()
    for x in y_array:
        a.add(x)
    return len(a)
    
###############################################################################    
def read_testdata(data_file,params,StateQueue):
    import pandas as pd
    from sklearn.preprocessing import normalize, minmax_scale
    
    #load test data
    data = pd.read_csv(data_file)

    X = data.drop('label',axis = 1)
    X = X.drop('SampleDate',axis = 1)
    X = X.drop('picktime',axis = 1)
    X = X.drop('classes',axis = 1)
    X = X.drop('temp_label',axis = 1)    
    y = data.label
    print "data.label :y",y.shape

    for wave in X.columns:
##        print wave
        if float(wave) <= params['wlengths'] or float(wave) >= params['wlengthe']:
            X = X.drop(wave,axis = 1)

    X,y = X[y != params['droplabel']],y[y != params['droplabel']]
##    print "X,y",X_new.shape,y.shape
    lambdstr = "lambda " + params['lambdastr']
    print "lambdstr",lambdstr
    X = X.apply(eval(lambdstr))
        
    if params['transform'] == 'none':
        print "transform = none"
        return X,y
    elif params['transform'] == 'norm':
        print "transform = norm"
        X = normalize(X, norm='l2')  
    elif params['transform'] == 'minmax':
        print "transform = minmax"
        X = minmax_scale(X)
    StateQueue.put("read test data done.")   
    return X, y

def read_data(filesAll,preproAll,boolAll,StateQueue):
    import pandas as pd
    import numpy as np
    import random
    import tensorflow as tf
    from sklearn.preprocessing import normalize, minmax_scale

    ##load train_data
    data = pd.read_csv(filesAll.getTrainDataFile())

    ##drop info collumn
    X = data.drop('label',axis = 1)
    X = X.drop('SampleDate',axis = 1)
    X = X.drop('picktime',axis = 1)
    X = X.drop('classes',axis = 1)
    X = X.drop('temp_label',axis = 1)    
    y = data.label
    print "data.label :y",y.shape

    ##drop label lambda 
    X,y = X[y != preproAll.getDropLabel()],y[y != preproAll.getDropLabel()]

    labelnum = staticLabelNum(y)
    preproAll.setLabelsNum(labelnum)
    ##select columns matched the wavelength
    for wave in X.columns:
##        print wave
        wlengths,wlengthe = preproAll.getWave()
        if float(wave) <= wlengths or float(wave) >= wlengthe:
            X = X.drop(wave,axis = 1)  
        
    ##apply the lambda fuction on data  
    lambdastr = "lambda " + preproAll.getLamStr()
    StateQueue.put("Shuffle %s" % boolAll.getShuffle())
    StateQueue.put("Apply %s" % boolAll.getApplyOr())
    print "lambda function is:",lambdastr
    if boolAll.getApplyOr():
        X = X.apply(eval(lambdastr)) 
    
    Data_new = np.c_[X , y]  
    print "########****Data_new",Data_new
    
    ## randomly shuffle data    
    if boolAll.getShuffle():
        random.shuffle(Data_new)
        
    ##cut Data_new to train and validate data 
    rate = preproAll.getDataRate()
    train = Data_new[:int(len(Data_new)*rate)]  
    test = Data_new[int(len(Data_new)*rate):]

    train_y = train[:,-1:]  
    train_x = train[:,:-1]

    test_y = test[:,-1:]  
    test_x = test[:,:-1]
    print "########****train_y",train_y.T
    ## onehot encode the train_y
    trainy_onehot = tf.one_hot(np.asarray(train_y.T)[0] , 4)
    
    if preproAll.getTransF() == 'none':
        print "transform = none"
        return train_x, train_y, test_x, test_y
    elif preproAll.getTransF() == 'norm':
        print "transform = norm"
        train_x = normalize(train_x, norm='l2')  
        test_x = normalize(test_x, norm='l2')
    elif preproAll.getTransF() == 'minmax':
        print "transform = minmax"
        train_x = minmax_scale(train_x)
        test_x = minmax_scale(test_x)

    StateQueue.put("read data done.")

    print "########****trainy_onehot",trainy_onehot.eval()
#    return train_x, trainy_onehot.eval(), test_x, test_y 
    return train_x, train_y, test_x, test_y

def SaveModelParam(filesAll,preproAll,ANNNET):
    ##自动保存训练好的模型，主要用于长时间训练参数无人值守用
    from sklearn.externals import joblib ##保存模型
    import pickle
    joblib.dump(filesAll.getModelName(),"..//DeepLearnNIR1.0//train"+"automodelsave.txt")
    wlengths,wlengthe = preproAll.getWave()   
    params = {'dimention':ANNNET.getOutDimention(),
              'wlengths':wlengths,
              'wlengthe':wlengthe,
              'transform':preproAll.getTransF(),
              'droplabel':preproAll.getDropLabel(),
              'lambdastr':preproAll.getLamStr()}

    f=file("..//DeepLearnNIR1.0//train//" + 'autoAllparams.txt','wb+')   
    pickle.dump(params,f)      
    f.close()     
