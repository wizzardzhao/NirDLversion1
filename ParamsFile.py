# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:02:09 2017

@author: liang
"""

class FileNameParams:
    import time
    ##训练数据文件区：    
    __trainfilepath =  "..//DeepLearnNIR1.0//train//" 

    __traindata_name = "..//DeepLearnNIR1.0//train//multi_apple_union.csv"
    
    __model_name = ''


    def getTrainFilePath(self):
        return self.__trainfilepath
    
    def setTrainDataFileName(self,filename):
        self.__traindata_name = filename
    def getTrainDataFile(self):
        print "Traindataname :",self.__traindata_name
        return self.__traindata_name
    
    def setModelName(self,modelname):
        self.__model_name = modelname
    def getModelName(self):
        return self.__model_name

    def getAllModelFile(self):
        return self.__trainfilepath + self.__allmodel_file
    

###########################################################
    ##测试过程文件名：

    __testdata_name = ''
    __testmodel_name = ''
    
    
    def setTestDataFileName(self,testdataname):
        self.__testdata_name = testdataname
    def getTestDataFile(self):
        print "datafilename :",self.__testdata_name
        return self.__testdata_name

    def setOpenModelName(self,openmodelname):
        self.__testmodel_name = openmodelname
    def getOpenModel(self):
        return self.__testmodel_name
        
class BoolFuncParams:
    __Shuffle = False     ##随机化数据
    __Write2File = False  ##是否输出到文件
    __ApplyOr = False   ##是否进行数据变换操作，与lambda关联

    def setShuffle(self,Shuffle):
        self.__Shuffle = Shuffle
    def getShuffle(self):
        return self.__Shuffle

    def setWrite2File(self,Write2File):
        self.__Write2File = Write2File
    def getWrite2File(self):
        return self.__Write2File
 
    def setApplyOr(self,ApplyOr):
        self.__ApplyOr = ApplyOr
    def getApplyOr(self):
        return self.__ApplyOr

##数据预处理
class PreProParams:
##    from Tkinter import *

    __lambdaStr = ''

    __StartWave = 490
    __EndWave = 1052
    
#    __outputDimention = 500    
    
    __positive_label = 4
    __droplabel = 9
    
    __labelsNum = 0
    
    __datarate = 0.7

    __BiOrMultiMode = 'micro'          ##分类模式选择
    
    __transform = 'None'      ##数据转换模式选择

    def setLamStr(self,lamstring):
        self.__lambdaStr = lamstring
    def getLamStr(self):
        return self.__lambdaStr

    def setWave(self,wlengths,wlengthe):
        if not isinstance(wlengths, int):
            raise ValueError('wlengths must be an integer!')
        if wlengths < 489 or wlengths > 1053:
            raise ValueError('wlengths must between 490 ~ 1052!')
        if wlengths > wlengthe :
            raise ValueError('wlengths must smaller than wlengthe!')
        if not isinstance(wlengthe, int):
            raise ValueError('wlengthe must be an integer!')
        if wlengthe < 489 or wlengthe > 1053:
            raise ValueError('wlengthe must between 490 ~ 1053!')
        self.__StartWave = wlengths
        self.__EndWave = wlengthe
        
    def getWave(self):
        return self.__StartWave,self.__EndWave

    def setBiOrMultiMode(self,BoM):
        self.__BiOrMultiMode = BoM
    def getBiOrMultiMode(self):
        return self.__BiOrMultiMode


    def setPosiLabel(self,plabel):
        if not isinstance(plabel, int):
            raise ValueError('plabel must be an integer!')
        if plabel not in [1,2,3,4] :
            raise ValueError('plabel must be {1,2,3,4}!')
        self.__positive_label = plabel
    def getPosiLabel(self):
        return self.__positive_label
    
    def setDropLabel(self,dlabel):
        if not isinstance(dlabel, int):
            raise ValueError('dlabel must be an integer!')
        if dlabel not in [1,2,3,4,5]:
            raise ValueError('dlabel must be {1,2,3,4}!')
        self.__droplabel = dlabel
    def getDropLabel(self):
        return self.__droplabel
        
    def setLabelsNum(self,labelnum):
        self.__labelsNum = labelnum
    def getLabelsNum(self):
        return self.__labelsNum

    def setDataRate(self,datar):
        if not isinstance(datar, float):
            raise ValueError('DataR must be an float!')
        if datar < 0 or datar > 1:
            raise ValueError('DataR must must between 0 ~ 1!')
        self.__datarate = datar
    def getDataRate(self):
        return self.__datarate

    def setTransF(self,transf):
        self.__transform = transf
    def getTransF(self):
        return self.__transform

    def printParams(self):
        print "dimen: %d DropLabel: %d datarate: %f" %(self.__SLFeaDimention, self.__droplabel,self.__datarate)
        print "transform :%s ,wlengths :%d,wlengthe : %d" %(self.__transform , self.__StartWave, self.__EndWave)
        print "featureMode :%s & featureInfo :%s" % (self.__featureMode,self.__selectedfeature_params)

        
        
class ROCParams:
    import numpy as np
    __mean_tpr = 0.0  
    __mean_fpr = np.linspace(0, 1, 100)  
    __all_tpr = []
    
    __thresh = 0.5

    def setMean_tpr(self,mean_tpr):
        self.__mean_tpr = mean_tpr
    def getMean_tpr(self):
        return self.__mean_tpr
    
    def interPMean_tpr(self, fpr, tpr):
        from scipy import interp
        self.__mean_tpr += interp(self.__mean_fpr, fpr, tpr)          #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数  
        self.__mean_tpr[0] = 0.0                               #初始处为0

    def setMean_fpr(self,mean_fpr):
        self.__mean_fpr = mean_fpr
    def getMean_fpr(self):
        return self.__mean_fpr

class TrainData:
    __x = []
    __y = []

    def setTrainX(self,X):
        self.__x = X
    def getTrainX(self):
        return self.__x

    def setTrainY(self,y):
        self.__y = y
    def getTrainY(self):
        return self.__y
    
    def printTrainDataInfo(self):
        print "trainX , trainY :",self.__x.shape,self.__y.shape
    
class TestData:
    __test_x = []
    __test_y = []

    def setTestY(self,y):
        self.__test_y = y
    def getTestY(self):
        return self.__test_y
    
    def setTestX(self,X):
        self.__test_x = X
    def getTestX(self):
        return self.__test_x
    
    def printTestDataInfo(self):
        print "testX , testY :",self.__test_x.shape,self.__test_y.shape

class Result:
    __predict = []
    __precision = 0.0
    __recall = 0.0
    __accuracy = 0.0

    def setPredict(self,predict):
        self.__predict = predict
    def getPredict(self):
        return self.__predict
    
    def setPrecision(self,precision):
        self.__precision = precision
    def getPrecision(self):
        return self.__precision
    
    def setRecall(self,recall):
        self.__recall = recall
    def getRecall(self):
        return self.__recall

    def setAccuracy(self,accuracy):
        self.__accuracy = accuracy
    def getAccuracy(self):
        return self.__accuracy    
