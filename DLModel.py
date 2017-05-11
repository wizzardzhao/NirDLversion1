# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:25:01 2017

@author: liang
"""
class DeepLearnTrainParams:
    __step1Learningrate = 0.0
    __step1_DL_Iterations = 0
    __step1_DL_Batchsize = 0

    __step2_DL_Learning_rate = 0.0
    __step2_DL_Iterations = 0
    __step2_DL_Batchsize = 0
    
    __step3_DL_Learning_rate = 0.0
    __step3_DL_Iterations = 0
    __step3_DL_Batchsize = 0
    
    def __init__(self):
        self.__step1_DL_Learning_rate = 0.01
        self.__step1_DL_Iterations = 12000
        self.__step1_DL_Batchsize = 40
    
        self.__step2_DL_Learning_rate = 0.001
        self.__step2_DL_Iterations = 6000
        self.__step2_DL_Batchsize = 35
        
        self.__step3_DL_Learning_rate = 0.001
        self.__step3_DL_Iterations = 3000
        self.__step3_DL_Batchsize = 30
        
    def setDLLearningRate(self,st1_rate,st2_rate,st3_rate):
        self.__step1_DL_Learning_rate = st1_rate
        self.__step2_DL_Learning_rate = st2_rate
        self.__step3_DL_Learning_rate = st3_rate
    def getDLLearningRate(self):
        return self.__step1_DL_Learning_rate,self.__step2_DL_Learning_rate,self.__step3_DL_Learning_rate
        
    def setDLIterations(self,st1_iter,st2_iter,st3_iter):
        self.__step1_DL_Iterations = st1_iter
        self.__step2_DL_Iterations = st2_iter
        self.__step3_DL_Iterations = st3_iter
    def getDLIterations(self):
        return self.__step1_DL_Iterations,self.__step2_DL_Iterations,self.__step3_DL_Iterations
        
    def setDLBatchsize(self,st1_batch,st2_batch,st3_batch):
        self.__step1_DL_Batchsize = st1_batch
        self.__step2_DL_Batchsize = st2_batch
        self.__step3_DL_Batchsize = st3_batch
    def getDLBatchsize(self):
        return self.__step1_DL_Batchsize,self.__step2_DL_Batchsize,self.__step3_DL_Batchsize
        
class ANNETModel:
    __activations = ''
    __regularization = ''
    __optimization = ''
    __initmode = ''
    __regularzationrate = 0.0
    __dropoutrate = 0.0
    __layersnum = 0
    __sigma = 0.0
    __outputdimention = 0
    __modelname = ''    
    
    def __init__(self):
        self.__activations = 'relu'
        self.__regularization = 'l1'
        self.__optimization = 'SGD'
        self.__initmode = 'uniform'
        self.__regularzationrate = 0.0
        self.__dropoutrate = 0.0
        self.__layersnum = 3
        self.__sigma = 0.0
        self.__outputdimention = 100
        
    def setACTFUNC(self,actfunc):
        self.__activations = actfunc
    def getACTFUNC(self):
        return self.__activations
        
    def setInit(self,initmode):
        self.__initmode = initmode
    def getInit(self):
        return self.__initmode
        
    def setOutDimention(self,dimention):
        self.__outputdimention = dimention
    def getOutDimention(self):
        return self.__outputdimention
        
    def setLayersNum(self,layersnum):
        self.__layersnum = layersnum
    def getLayersNum(self):
        return self.__layersnum        
        
    def setRegularFUNC(self,regularfunc):
        self.__regularization = regularfunc
    def getRegularFUNC(self):
        return self.__regularization
        
    def setRegularRate(self,regularrate):
        self.__regularzationrate = regularrate
    def getRegularRate(self):
        return self.__regularzationrate
        
    def setOPTFUNC(self,optfunc):
        self.__optimization = optfunc
    def getOPTFUNC(self):
        return self.__optimization
        
    def setDropoutRate(self,dropoutrate):
        self.__dropoutrate = dropoutrate
    def getDropoutRate(self):
        return self.__dropoutrate
        
    def setGNsigma(self,sigma):
        self.__sigma = sigma
    def getGNsigma(self):
        return self.__sigma   

    def setAnnModelname(self,model):
        self.__modelname = model
    def getAnnModelname(self):
        return self.__modelname         
    
    # create model BP-net
    def ANN_Custom(self,inputdimention,labelsnum):
        from keras.models import Sequential
        from keras.layers import Dense 
        from keras.layers import Dropout
        from keras.layers.noise import GaussianNoise
        from keras.constraints import maxnorm
        
        self.__model = Sequential()
        
        self.__model.add(Dense(output_dim = inputdimention, input_dim=inputdimention, 
                        init=self.getInit(),
                        activation=self.getACTFUNC(), 
                        W_constraint=maxnorm(1))) #2构建图模型  
        #add noise layer
        self.__model.add(GaussianNoise(self.getGNsigma()))
        
        
        ##hidden layer
        outputdimention = self.getOutDimention()
        
        layersNum = self.getLayersNum()
        
        subdim = int((inputdimention - outputdimention)/layersNum)
        print "subdim",subdim
        while layersNum > 0:
            inputdim = inputdimention - subdim*(self.getLayersNum() - layersNum)
            outputdim = inputdim
            print "outputdim",outputdim
            self.__model.add(Dense(output_dim = outputdim,input_dim = inputdim,
                            init=self.getInit(),
                            activation = self.getACTFUNC(),
                            W_constraint=maxnorm(1)))
            self.__model.add(Dropout(self.getDropoutRate()))
            layersNum = layersNum-1
            print "layersNum--",layersNum
            
        ##full-conect layer            
        self.__model.add(Dense(output_dim=outputdimention, input_dim=outputdimention,
            init=self.getInit(),
            activation = self.getACTFUNC(),
            W_constraint=maxnorm(1)))
    
        ##last layer
        self.__model.add(Dense(output_dim = labelsnum, input_dim=outputdimention, 
                        init = self.getInit(), 
                        activation='softmax',
                        W_constraint=maxnorm(1)))#,W_constraint=maxnorm(1)
    
        return self.__model    

    def ANN_simple2(self):
        from keras.models import Sequential
        from keras.layers.core import Activation
        from keras.layers import Dense 
        from keras.layers import Dropout
        from keras.layers.noise import GaussianNoise
        from keras.constraints import maxnorm
        
        model = Sequential()
    
        model.add(Dense(output_dim=950, input_dim=1173, 
                        init=self.getInit(),
                        activation=self.getACTFUNC(), 
                        W_constraint=maxnorm(1))) #2构建图模型  
        #add noise layer
        model.add(GaussianNoise(0.01))
    
        model.add(Dense(output_dim=750, input_dim=850,
                        init=self.getInit(),
                        activation=self.getACTFUNC(), 
                        W_constraint=maxnorm(1)))
        model.add(Dropout(0.2))
    
        ##
        model.add(Dense(output_dim=500, input_dim=550,
                        init=self.getInit(),
                        activation=self.getACTFUNC(), 
                        W_constraint=maxnorm(1)))
        ##last layer 
        model.add(Dropout(0.1))
        model.add(Dense(output_dim = 4, input_dim=450, 
                        init=self.getInit(),
                        activation='softmax', 
                        W_constraint=maxnorm(1)))#,W_constraint=maxnorm(1)
    
        return model    
        
    def ANN_GridSearch(self):
        from keras.models import Sequential
        from keras.layers import Dense 
        from keras.layers import Dropout
        from keras.layers.noise import GaussianNoise
        from keras.constraints import maxnorm
        
        from sklearn.grid_search import GridSearchCV

        from keras.wrappers.scikit_learn import KerasClassifier
        
        model = Sequential()
    
        model.add(Dense(output_dim=950, input_dim=1173, 
                        init=self.getInit(),
                        activation=self.getACTFUNC(), 
                        W_constraint=maxnorm(1))) #2构建图模型  
        #add noise layer
        model.add(GaussianNoise(0.01))
    
        model.add(Dense(output_dim=750, input_dim=850,
                        init=self.getInit(),
                        activation=self.getACTFUNC(), 
                        W_constraint=maxnorm(1)))
        model.add(Dropout(0.2))
    
        ##
        model.add(Dense(output_dim=500, input_dim=550,
                        init=self.getInit(),
                        activation=self.getACTFUNC(), 
                        W_constraint=maxnorm(1)))
        ##last layer 
        model.add(Dropout(0.1))
        model.add(Dense(output_dim = 4, input_dim=450, 
                        init=self.getInit(),
                        activation='softmax', 
                        W_constraint=maxnorm(1)))#,W_constraint=maxnorm(1)
    
        return model
 
