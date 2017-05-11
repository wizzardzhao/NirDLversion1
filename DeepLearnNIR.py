# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:13:18 2017

@author: liang
"""

import time,threading,Queue  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Button ##添加按钮功能包
from Tkinter import *
import ttk
##import tkinter
import tensorflow as tf
from sklearn.externals import joblib ##保存模型

import DLModel 
import ParamsFile

def saveModel_press(model,StateQueue):
    import tkFileDialog
    import pickle
    import h5py
    import keras.models
    
    temp_modelfile = ''
    direct = "..//DeepLearnNIR1.0//model//" 
    save_modelname = 'ANNnet.json'   
    temp_modelfile = tkFileDialog.asksaveasfilename(title=u'保存模型文件', initialdir = direct, initialfile = save_modelname)
    print(temp_modelfile)


    #保存json文件
    json_string = model.to_json()  
    open(temp_modelfile,'wb+').write(json_string)

    model.save('ANNnet.h5')
    model.save_weights(direct+'ANN_model_weights.h5',overwrite=True)
    
    wlengths,wlengthe = preproAll.getWave()   
    params = {##'dimention':preproAll.getFeaDimen(),
              'wlengths':wlengths,
              'wlengthe':wlengthe,
              'transform':preproAll.getTransF(),
              'droplabel':preproAll.getDropLabel(),
              'lambdastr':preproAll.getLamStr()}
    ##把字典内容存入文件
    f=file("..//DeepLearnNIR1.0//model//" + 'params.txt','wb+')   #新建文件data.txt，'wb'，b是打开块文件，对于设备文件有用
    pickle.dump(params,f)      #把a序列化存入文件
    f.close()
    print "Params are saved!"
    print '%s is saved!' % save_modelname
    StateQueue.put("Params are saved! %s is saved!" % save_modelname)

def shufflebutton_press(shuf):
    if shuf.get() == 1:
        boolAll.setShuffle(True)  
    else:
        boolAll.setShuffle(False) 

def ApplyOr_press(lam):
    if lam.get() == 1:
        boolAll.setApplyOr(True)  
    else:
        boolAll.setApplyOr(False)

def Writetofile_press(wri):
    if wri.get() == 1:
        boolAll.setWrite2File(True)
        print "predict结果写出到文件"
    else:
        boolAll.setWrite2File(False) 

    
def readtestdata_press(StateQueue):
    import tkFileDialog,pickle
    import FileMethod
    direct = "..//DeepLearnNIR1.0//"
        
    temp = open(direct + 'model//params.txt','rb')    #默认打开训练参数文件  a=open('data.txt','rb')    #打开文件
    print "temp",temp

    params = pickle.load(temp)      #把内容全部反序列化
    print "params test:",params
    StateQueue.put("test params is:")
    StateQueue.put(params)
    test_data_file = tkFileDialog.askopenfilename(title=u'打开测试文件',initialdir = direct+'test', filetypes=[('csv', '*.csv'), ('All Files', '*')])#,initialdir = direct

    print "testfile:***********",test_data_file
    filesAll.setTestDataFileName(test_data_file)
    tempread = FileMethod.read_testdata(test_data_file,params,StateQueue)
    testData.setTestX( tempread[0])
    testData.setTestY( tempread[1])    
    testData.printTestDataInfo()
    StateQueue.put("test file open success!")
    print"test file open success!"


def readnewtrainfile_press(StateQueue):
    import tkFileDialog
    data_file = ''
    direct = "..//DeepLearnNIR1.0//train//"
    data_file = tkFileDialog.askopenfilename(title=u'打开训练文件',initialdir = direct, filetypes=[('csv', '*.csv'), ('All Files', '*')])#,initialdir = direct
    filesAll.setTrainDataFileName(data_file)
    StateQueue.put("read file")
    StateQueue.put(data_file)
    StateQueue.put("train file open success!")
    print(data_file)
    

def openmodel_press(StateQueue):
    import tkFileDialog

    direct = "..//DeepLearnNIR1.0//model//"
    openmodelname = 'ANNmodel.json'
    openmodelfile = tkFileDialog.askopenfilename(title=u'打开保存模型', initialdir = direct,initialfile = openmodelname,filetypes=[('json', '*.json'), ('All Files', '*')])#initialdir = direct , 

    #读取model
    StateQueue.put("#############################")
    StateQueue.put("Read model")

    ANNNET.setAnnModelname(openmodelfile)

    StateQueue.put("%s model load success!" %openmodelfile)
    print(openmodelname)    
    print"model load success!"

def predict_classes(proba):
    '''Generate class predictions for the input samples
    batch by batch.

    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    # Returns
        A numpy array of class predictions.
    '''
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')
        
def probfunction(proba):
    array_proba=[]
    j=0
    while j<proba.shape[0]:
        i=proba[j].max()
        array_proba.append(i)
        j+=1
    return array_proba
        
def predict_press(predict_flag,StateQueue):
    from sklearn import metrics

    from keras.models import model_from_json
    from keras.models import load_model
    import h5py
    import numpy as np
    import FileMethod    
    
    direct = "..//DeepLearnNIR1.0//model//"
    print "start predict"
    StateQueue.put("######################################")
    gui.insertTextBox()
    StateQueue.put("*start predict")
    gui.insertTextBox()
    print "OpenModel" , filesAll.getOpenModel()
    
    openmodelname = ANNNET.getAnnModelname()
    
    model = model_from_json(open(openmodelname).read())  
    model.load_weights(direct+'ANN_model_weights.h5')

##    model = load_model('ANNnet.h5')
    print "model is:",model
    predict = model.predict_classes(testData.getTestX())

    
    #probilities for all labels
    predictprob = model.predict_proba(testData.getTestX())
    print predictprob.max()
    print predictprob.shape 
    predictproba= probfunction(predictprob)
#    x = predict_classes(predictprob)
    print "XXXXXXX",predictproba

    print "predict shape is: ",predict.shape
    
    proba=[]
    proba = np.c_[testData.getTestY(),predict]
    proba = np.c_[proba,predictproba]
    StateQueue.put("predict value is:" )
    gui.insertTextBox()
    StateQueue.put(predict)
    StateQueue.put(np.array(proba))
    gui.insertTextBox()
    print "predict value is:",predict
    print "predict value is:",proba
    
    precision = metrics.precision_score(testData.getTestY(), predict, average = preproAll.getBiOrMultiMode())    
    recall = metrics.recall_score(testData.getTestY(), predict,average = preproAll.getBiOrMultiMode())    
    print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    StateQueue.put('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    gui.insertTextBox()
    accuracy = metrics.accuracy_score(testData.getTestY(), predict)    
    print('accuracy: %.2f%%' % (100 * accuracy))
    StateQueue.put('accuracy: %.2f%%' % (100 * accuracy))
    gui.insertTextBox()
    f1 = metrics.f1_score(testData.getTestY(),predict,average = 'weighted')
    StateQueue.put('f1 score: %.4f' % (f1))
    gui.insertTextBox()
    print('f1 score: %.4f' % (f1))  
    
    if boolAll.getWrite2File():
        print "test data:",filesAll.getTestDataFile()
        FileMethod.AddLabels(filesAll.getTestDataFile(),predict,'predict')
    StateQueue.put("predict success")
    gui.insertTextBox()
    gui.predict_flag = True

    print "predict success"   

def _async_raise(tid, exctype):
    import inspect
    import ctypes
    
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

class MainGui():
    def __init__(self,master):
        self.master=master  
        self.StateQueue = Queue.Queue()
        self.InfoQueue = Queue.Queue()
        self.PredictQueue = Queue.Queue()
        self.train_flag = False
        
        ##定义功能区：三大块
        ##上部：菜单行**********************************************************************************************************************
        self.frm_top = LabelFrame(root)
        self.N_btn = Button(self.frm_top,text='Train File',command = lambda:readnewtrainfile_press(self.StateQueue))
        self.N_btn.grid(row=0,column=0,columnspan=1,sticky = W)
        
        self.f_btn = Button(self.frm_top,text='Test File',command =lambda:readtestdata_press(self.StateQueue))
        self.f_btn.grid(row=0,column=1,columnspan=1,sticky = W) #
        
        self.l_btn = Button(self.frm_top,text='Load Model',command = lambda:openmodel_press(self.StateQueue))
        self.l_btn.grid(row=0,column=2,columnspan=1,sticky = W)
        
#        self.SS_btn = Button(self.frm_top,text="Save Model",command = lambda :saveModel_press(self.StateQueue))
#        self.SS_btn.grid(row=0,column=3,columnspan=1,sticky = W)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.p_btn = Button(self.frm_top,text='Predict',command = self.startPredict)
            self.p_btn.grid(row=0,column=5,columnspan=1,sticky = W)
        
        self.dm_btn = Button(self.frm_top,text="Report",command = None)
        self.dm_btn.grid(row=0,column=7,columnspan=1,sticky = W)

        self.wri = IntVar()
        self.Write2File = Checkbutton(self.frm_top,
                                 text='Write',
                                 variable = self.wri,
                                 command = lambda:Writetofile_press(self.wri))
        self.Write2File.grid(row=0,column=4,columnspan=1, sticky = W)
        
        self.frm_top.grid(row= 0,column = 0,columnspan=18,sticky ='WESN')
        
        ##中间：图像结果显示行**************************************************************************************************************************
    ##    frm_middle_left = LabelFrame(root).grid(row= 1,column = 0,sticky ='WESN')
        #在frm_middle_left的GUI上放置一个画布，并用.grid()来调整布局

        self.fig = Figure(figsize = [7.5,6],dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master = root)
        self.fig_sub = self.fig.add_subplot(111)
        DrawPic(self.canvas,self.fig_sub)
        
        self.frm_middle_right = LabelFrame(root)
        self.text_box = Text(self.frm_middle_right,width=50,heigh = 42)     

        self.text_box.grid(row = 0,column = 0)
        self.text_box.insert('end',"Hello!")
        self.scrollY_0 = Scrollbar(self.frm_middle_right,orient=VERTICAL,command = self.text_box.yview)
        self.text_box['yscrollcommand'] = self.scrollY_0.set
        self.scrollY_0.grid(row= 0,column = 6,sticky = 'N'+'S'+'E'+'W')
        
        self.frm_middle_right.grid(row= 1,column = 4,columnspan=9,sticky ='WESN')    

        ##下部参数设置以及训练控制行*********************************************************************************************************************
        self.frm_bottom1 = LabelFrame(root)
        
        ##1*************************** data preprocess****************************        
        self.LF_datapreprocess = LabelFrame(self.frm_bottom1,text = 'NIR Data Preprocess')
        self.D_lbl = Label(self.LF_datapreprocess,text="DropL:")
        self.D_lbl.grid(row=0,column=0,sticky = W)
        self.inputLabel=Entry(self.LF_datapreprocess,width = 4)
        self.inputLabel.grid(row=0,column=1,sticky = E)
        self.inputLabel.insert(0,'5')

        self.P_lbl = Label(self.LF_datapreprocess,text="PosiL:")
        self.P_lbl.grid(row=0,column=2,sticky = W)
        self.posiLabel=Entry(self.LF_datapreprocess,width = 4)
        self.posiLabel.grid(row=0,column=3,sticky = E)
        self.posiLabel.insert(0,'3')

        Label(self.LF_datapreprocess,text="ClsMode:").grid(row=0,column=4,columnspan=1)
        self.clsM = StringVar()
        self.BiOrMultiMode = ttk.Combobox(self.LF_datapreprocess, width=8, textvariable = self.clsM)
        self.BiOrMultiMode['values'] = ('macro','binary','micro')     # 设置下拉列表的值
        self.BiOrMultiMode.grid(row=0,column=5)      # 设置其在界面中出现的位置  column代表列   row 代表行
        self.BiOrMultiMode.insert(0,'macro')    # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值
        
        Label(self.LF_datapreprocess,text="TraForm:").grid(row=1,column=4,columnspan=1,sticky = W)
        self.trans = StringVar()
        self.inputtransform = ttk.Combobox(self.LF_datapreprocess, width=8, textvariable = self.trans)
        self.inputtransform['values'] = ('norm','minmax','none')     # 设置下拉列表的值
        self.inputtransform.grid(row=1,column=5,sticky = W)      # 设置其在界面中出现的位置  column代表列   row 代表行
        self.inputtransform.insert(0,'norm')    # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值

        self.lam = IntVar()
        self.Apply = Checkbutton(self.LF_datapreprocess,text='Apply(SpectraData):',variable = self.lam,command =lambda:ApplyOr_press(self.lam))
        self.Apply.grid(row=0,column=6,sticky = W)     
        self.inputLambda=Entry(self.LF_datapreprocess,width = 12)
        self.inputLambda.grid(row=0,column=7,columnspan=1,sticky = W)
        self.inputLambda.insert(0,'x:x*1')
       

        ##2***************************2****************************************************
        Label(self.LF_datapreprocess,text="WLs:").grid(row=1,column=0,sticky = W)
        self.inputwlengths=Entry(self.LF_datapreprocess,width = 4)
        self.inputwlengths.grid(row=1,column=1,sticky = E)
        self.inputwlengths.insert(0,'489')

        Label(self.LF_datapreprocess,text="WLe:").grid(row=1,column=2,sticky = W)
        self.inputwlengthe=Entry(self.LF_datapreprocess,width = 4)
        self.inputwlengthe.grid(row=1,column=3,sticky = W)
        self.inputwlengthe.insert(0,'1053')
        
        self.shuf = IntVar()
        self.shufflebutton = Checkbutton(self.LF_datapreprocess,
                                    text='Shuffle',
                                    variable = self.shuf,
                                    command = lambda:shufflebutton_press(self.shuf))
        self.shufflebutton.grid(row=1,column=6,columnspan=1, sticky=W)
        print "befor first shuffe: ",filesAll.getTrainDataFile()
        
        self.rate = IntVar()
        Label(self.LF_datapreprocess,text="DataR:").grid(row=1,column=7,columnspan=1,sticky = W)   
        self.inputRate=Entry(self.LF_datapreprocess,width = 6)
        self.inputRate.grid(row=1,column=7,columnspan=1,sticky = E)
        self.inputRate.insert(0,'0.7') 
        
        self.LF_datapreprocess.grid(row= 0,column = 0,columnspan=8,sticky ='WESN')
        
        ##******************************model params***************************************************
        self.LF_modelparams = LabelFrame(self.frm_bottom1,text = 'DL Model Params')
        
        Label(self.LF_modelparams,text="ACT_Func:").grid(row=0,column=0,sticky = W)
        self.str_act = StringVar()
        self.DL_ActFunc = ttk.Combobox(self.LF_modelparams, width=8, textvariable = self.str_act)
        self.DL_ActFunc['values'] = ('relu','sigmoid','tanh','linear')     # 设置下拉列表的值
        self.DL_ActFunc.grid(row=0,column=1,sticky = E)      # 设置其在界面中出现的位置  column代表列   row 代表行
        self.DL_ActFunc.insert(0,'relu')    # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值
        

        Label(self.LF_modelparams,text="Optimization:").grid(row=0,column=4,columnspan=1,sticky = W)
        self.opt = StringVar()
        self.DL_Optimize = ttk.Combobox(self.LF_modelparams, width=8, textvariable = self.opt)
        self.DL_Optimize['values'] = ('SGD','Adam','RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam')     # 设置下拉列表的值
        self.DL_Optimize.grid(row=0,column=5,sticky = E)      # 设置其在界面中出现的位置  column代表列   row 代表行
        self.DL_Optimize.insert(0,'SGD')    # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值
        
        Label(self.LF_modelparams,text="GN Sigma:").grid(row=0,column=6,columnspan=1,sticky = W)
        self.sigma = StringVar()
        self.DL_GNsigma = ttk.Combobox(self.LF_modelparams, width=8, textvariable = self.sigma)
        self.DL_GNsigma['values'] = ('0','0.1','0.05', '0.01', '0.005')     # 设置下拉列表的值
        self.DL_GNsigma.grid(row=0,column=7,sticky = E)      # 设置其在界面中出现的位置  column代表列   row 代表行
        self.DL_GNsigma.insert(0,'0.01')    # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值

        Label(self.LF_modelparams,text="Layers:").grid(row=0,column = 8,sticky = W)
        self.DL_Model_layers=Entry(self.LF_modelparams,width = 6)
        self.DL_Model_layers.grid(row=0,column=9,columnspan=1,sticky = E)
        self.DL_Model_layers.insert(0,'3')
        
        self.uniform = StringVar()
        Label(self.LF_modelparams,text="Init Mode:").grid(row=1,column=0,columnspan=1,sticky = W)
        self.DL_Initmode = ttk.Combobox(self.LF_modelparams, width=8, textvariable = self.uniform)
        self.DL_Initmode['values'] = ('uniform','glorot_uniform')     
        self.DL_Initmode.grid(row =1, column=1,sticky = W)      
        self.DL_Initmode.insert(0,'uniform')    
        
        self.regularization = StringVar()
        Label(self.LF_modelparams,text="Regularization Func:").grid(row=0,column=2,columnspan=1,sticky = W)
        self.DL_Regularization = ttk.Combobox(self.LF_modelparams, width=8, textvariable = self.regularization)
        self.DL_Regularization['values'] = ('l1','l2','None')    
        self.DL_Regularization.grid(row =0, column=3,sticky = W)      
        self.DL_Regularization.insert(0,'l2')           
        

        self.regular_rate = StringVar()
        Label(self.LF_modelparams,text="Regularization Rate:").grid(row=1,column=2,columnspan=1,sticky = W)
        self.DL_Regularization_rate = ttk.Combobox(self.LF_modelparams, width=8, textvariable = self.regular_rate)
        self.DL_Regularization_rate['values'] = ('0','0.001','0.003','0.01','0.03','0.1','0.3','1','3','10')     
        self.DL_Regularization_rate.grid(row = 1, column=3,sticky = W)      
        self.DL_Regularization_rate.insert(0,'0.1')  
        

        Label(self.LF_modelparams,text="OutDimention:").grid(row=1,column=4,columnspan=1,sticky = W)
        self.DL_Model_outputdimention=Entry(self.LF_modelparams,width = 10)
        self.DL_Model_outputdimention.grid(row=1,column=5,columnspan=1,sticky = E)
        self.DL_Model_outputdimention.insert(0,'100')    

        self.dropout_rate = StringVar()
        Label(self.LF_modelparams,text="Dropout Rate:").grid(row=1,column=6,columnspan=1,sticky = W)
        self.DL_Model_droprate = ttk.Combobox(self.LF_modelparams, width=8, textvariable = self.dropout_rate)
        self.DL_Model_droprate['values'] = ('0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8')     
        self.DL_Model_droprate.grid(row = 1, column=7,sticky = W)      
        self.DL_Model_droprate.insert(0,'0.2')  
        
        ## weights=None, W_regularizer=None, b_regularizer=None, 
        ## activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None

        self.LF_modelparams.grid(row= 1,column = 0,columnspan=8,sticky ='WESN')
#        
        self.frm_bottom1.grid(row= 2,column = 0,columnspan=8,sticky ='WESN')

        ##下部参数设置以及训练控制行222222222*********************************************************************************************************************        
        self.frm_bottom2 = LabelFrame(root)
         
        ##1***************************step1****************************
        self.trainstep_1 = LabelFrame(self.frm_bottom2, text="Train_Step1 Params")              
        
        Label(self.trainstep_1,text="L_Rate:").grid(row=1,column = 0,sticky = W)
        self.step1_DL_Learning_rate=Entry(self.trainstep_1,width = 6)
        self.step1_DL_Learning_rate.grid(row=1,column=1,columnspan=1,sticky = E)
        self.step1_DL_Learning_rate.insert(0,'0.01')
        
        
        Label(self.trainstep_1,text="Iterations:").grid(row=1,column=2,columnspan=1,sticky = W)
        self.step1_Iterations = Entry(self.trainstep_1,width = 6)
        self.step1_Iterations.grid(row=1,column=3,columnspan=1,sticky = E)
        self.step1_Iterations.insert(0,'120')
        
        Label(self.trainstep_1,text="Batchsize:").grid(row=1,column=4,columnspan=1,sticky = W)
        self.step1_Batchsize=Entry(self.trainstep_1,width = 4)
        self.step1_Batchsize.grid(row=1,column=5,columnspan=1,sticky = W)
        self.step1_Batchsize.insert(0,'40')        
    
        self.trainstep_1.grid(row= 0,column = 0,columnspan=6,sticky ='WESN')
        
        ##1***************************step2********************************
        self.trainstep_2 = LabelFrame(self.frm_bottom2, text="Train_Step2 Params")
        
        Label(self.trainstep_2,text="L_Rate:").grid(row=1,column=0,sticky = W)
        self.step2_DL_Learning_rate=Entry(self.trainstep_2,width = 6)
        self.step2_DL_Learning_rate.grid(row=1,column=1,columnspan=1,sticky = E)
        self.step2_DL_Learning_rate.insert(0,'0.001')
        
        
        Label(self.trainstep_2,text="Iterations:").grid(row=1,column=2,columnspan=1,sticky = W)
        self.step2_Iterations=Entry(self.trainstep_2,width = 6)
        self.step2_Iterations.grid(row=1,column=3,columnspan=1,sticky = E)
        self.step2_Iterations.insert(0,'60')
        
        Label(self.trainstep_2,text="Batchsize:").grid(row=1,column=4,columnspan=1,sticky = W)
        self.step2_Batchsize=Entry(self.trainstep_2,width = 4)
        self.step2_Batchsize.grid(row=1,column=5,columnspan=1,sticky = W)
        self.step2_Batchsize.insert(0,'35')        
    
        self.trainstep_2.grid(row= 1,column = 0,columnspan=6,sticky ='WESN')
        
        ##1***************************step3********************************
        self.trainstep_3 = LabelFrame(self.frm_bottom2, text="Train_Step3 Params")
        
        Label(self.trainstep_3,text="L_Rate:").grid(row=1,column=0,sticky = W)
        self.step3_DL_Learning_rate=Entry(self.trainstep_3,width = 6)
        self.step3_DL_Learning_rate.grid(row=1,column=1,columnspan=1,sticky = E)
        self.step3_DL_Learning_rate.insert(0,'0.0001')
        
        
        Label(self.trainstep_3,text="Iterations:").grid(row=1,column=2,columnspan=1,sticky = W)
        self.step3_Iterations=Entry(self.trainstep_3,width = 6)
        self.step3_Iterations.grid(row=1,column=3,columnspan=1,sticky = E)
        self.step3_Iterations.insert(0,'30')
        
        Label(self.trainstep_3,text="Batchsize:").grid(row=1,column=4,columnspan=1,sticky = W)
        self.step3_Batchsize = Entry(self.trainstep_3,width = 4)
        self.step3_Batchsize.grid(row=1,column=5,columnspan=1,sticky = W)
        self.step3_Batchsize.insert(0,'30')        
    
        self.trainstep_3.grid(row= 2,column = 0,columnspan=6,sticky ='WESN')
        
        self.startbutton = Button(self.frm_bottom2,text="Run",command = self.startTrain,fg='blue')
        self.startbutton.grid(row=0,column=18,columnspan=1,sticky = 'WESN')#       

        self.stopbutton = Button(self.frm_bottom2,text="Stop",command = self.stop_trainthread,fg='red')
        self.stopbutton.grid(row=1,column=18,columnspan=1,sticky = 'WESN')#
        
        self.frm_bottom2.grid(row= 2,column = 4,columnspan=12,sticky ='WESN')
        

    def clearCanvas(self):
        self.fig.clf()
        self.canvas = FigureCanvasTkAgg(self.fig, master = root)
        self.fig_sub = self.fig.add_subplot(111)

    ##将每次训练任务放到一个独立的线程中进行，实现多线程
    def startTrain(self):
        refreshParam(preproAll,ANNNET,dltrainAll)
        self.__threadTrain=threading.Thread(target=self.trainmodel)
        self.train_flag = False
        self.__threadTrain.setDaemon(True)
        self.__threadTrain.start()
#        self.currentthread = self.__threadTrain.getName()
        if not self.train_flag :
            self.periodicTextCall()
        else:
            self.canvas.show()
            self.__threadTrain.stop()
            self.__threadTrain.join()
            self.__threadTrain.exit()
        
        return self.__threadTrain

    def stopTrain(self):
        self.StateQueue.put("train stopped.")
        self.train_flag = True
        threadTrain.stop()
        threadTrain.join()
        threadTrain.exit()
        
    def stop_trainthread(self):
        _async_raise(self.__threadTrain.ident, SystemExit)
        self.StateQueue.put("train stopped.")
        self.train_flag = True
        print 'train stopped.'

##    def startPredict(self):
##        self.predict_flag = False
##        threadPredict = threading.Thread(target = lambda:predict_press(self.predict_flag,self.StateQueue))
##        threadPredict.start()
##        if not self.predict_flag:
##            self.periodicTextCall()
##        else:
##            self.canvas.show()
##            threadPredict.join()

    def startPredict(self):
        self.predict_flag = False
        predict_press(self.predict_flag,self.StateQueue)
#        self.insertTextBox()
        self.canvas.show()

    ##用queue来进行多线程之间的信息传递，为什么要用while？再了解下
    def insertTextBox(self):
        while self.StateQueue.qsize():  
            try:  
                msg = self.StateQueue.get(0) ##队列的第一个值是最新的消息
                self.text_box.insert("insert",'\n')##换行
                self.text_box.insert("insert",msg)   
            except Queue.Empty:  
                pass


    ##定义一个周期更新GUI方法，查询训练状态以及操作状态信息的方法
    def periodicTextCall(self):  
        self.master.after(200,self.periodicTextCall)
        self.insertTextBox()

        ###############################################################################    
    def PlotAcryROC(self,classifier,test_y,predict,precision,recall,accuracy):         
        from sklearn.metrics import roc_curve ,auc ,f1_score
        
        print "PlotAcryROC called",precision,recall,accuracy
        print "posi label",preproAll.getPosiLabel()
        fpr, tpr, thresholds = roc_curve(test_y, predict ,pos_label = preproAll.getPosiLabel())
        ROC.interPMean_tpr(fpr, tpr)
        print "fpr, tpr",fpr, tpr
        roc_auc = auc(fpr, tpr)
        print "roc_auc",roc_auc
        
        self.fig_sub.plot(fpr, tpr, lw=1,label='%s (precision: %.2f%%,recall: %.2f%%, accuracy: %.2f%%)' % (classifier, precision*100,recall*100,accuracy*100))

        #画ROC折线，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来  
        self.fig_sub.plot(fpr, tpr, lw=1, label='ROC  %s (area = %0.3f)' % (classifier, roc_auc))
        
    
    def GridsearchCv(self,create_model):
        from sklearn.grid_search import GridSearchCV

        from keras.wrappers.scikit_learn import KerasClassifier
        # create model
        model = KerasClassifier(build_fn=create_model, verbose=0)
        # define the grid search parameters
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 50, 100]
        param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
        
        return grid

    def floatfunction(self,array):
        print array.shape

    def trainmodel(self):##subfigue
        import numpy as np
        from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
        import tensorflow as tf
        import MethMethod
        sess = tf.InteractiveSession()
        print "model train start"
        self.StateQueue.put("#######################################")
        self.StateQueue.put("*ANN model train start")
        import FileMethod
        self.clearCanvas()

        train_x,train_y,test_x,test_y = FileMethod.read_data\
                            (filesAll,preproAll, boolAll,self.StateQueue)
#        print "train_y",train_y.shape

##        preproAll.setLabelsNum(len(count))
        labelsnum = preproAll.getLabelsNum()
        onehottrain_y = tf.one_hot(np.asarray(train_y.T)[0] , labelsnum).eval()
        
#        print "onehot train_y",onehottrain_y
        
        optmodel = ANNNET.getOPTFUNC()
        print 'optmodel',optmodel
        lr1,lr2,lr3 = dltrainAll.getDLLearningRate()
        print "lr1,lr2,lr3",lr1,lr2,lr3
        nb1,nb2,nb3 = dltrainAll.getDLBatchsize()
        print "nb1,nb2,nb3",nb1,nb2,nb3
        it1,it2,it3 = dltrainAll.getDLIterations()
        print "it1,it2,it3",it1,it2,it3
        
        opt1 = eval(optmodel)(lr=lr1, decay=1e-6)
        opt2 = eval(optmodel)(lr=lr2, decay=1e-6)
        opt3 = eval(optmodel)(lr=lr3, decay=1e-6)
        
        inputdimention = train_x.shape[1]

        print '********labelsnum*******',labelsnum
        self.StateQueue.put("%d labels to be classified" % labelsnum)
        model = ANNNET.ANN_Custom(inputdimention,labelsnum)
#        model = ANNNET.ANN_simple2()
        start_time = time.time()
        
        print "step 1...."  
        self.StateQueue.put("step 1....")
##        self.StateQueue.put(sys.stdout)
        
        model.compile(loss='categorical_crossentropy', optimizer=opt1 ,metrics=['accuracy'])  ##loss:categorical_crossentropy  sparse_categorical_crossentropy,
#        model = self.GridsearchCv(model)
        model.fit(train_x, onehottrain_y, nb_epoch=it1, batch_size=nb1,shuffle = True,verbose=2) #4训练模型

#        grid_result = model.fit(train_x, train_y)
#        # summarize results
#        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#        for params, mean_score, scores in grid_result.grid_scores_:

#            print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
            
        print "step 2...."
        self.StateQueue.put("step 2....")
##        self.StateQueue.put(sys.stdout)
        model.compile(loss='categorical_crossentropy', optimizer=opt2 ,metrics=['accuracy'])
        model.fit(train_x, onehottrain_y, nb_epoch=it2, batch_size=nb2,shuffle = True,verbose=2) #4训练模型

        print "step 3...."
        self.StateQueue.put("step 3....")
##        self.StateQueue.put(sys.stdout)
        model.compile(loss='categorical_crossentropy', optimizer=opt3 ,metrics=['accuracy'])
        model.fit(train_x, onehottrain_y, nb_epoch=it3, batch_size=nb3,shuffle = True,verbose=2) #4训练模型
                
#        json_string = model.to_json()  
#        open('my_model_architecture.json','wb+').write(json_string)  
#        model.save_weights('ANN_model_weights.h5',overwrite=True)
#        print"train end!"
        
        self.StateQueue.put("######################################")
        self.StateQueue.put("model train end")
        costtime = time.time() - start_time
        print 'training took %fs!' % costtime        

        self.StateQueue.put('training took %fs!' % (costtime))
        self.StateQueue.put("%s train done." % model )


        predict = model.predict_classes(test_x)
        
     
        
        print "True label is",test_y.T
        print "predict is ",predict
        
        
#        print "predict score is %0.3f"%predictprob
        self.StateQueue.put("predict is %s" % predict )
        precision,recall,accuracy = \
            MethMethod.CalculateMixtureMetrics(test_y, predict, preproAll.getBiOrMultiMode())
        classifier = 'ANN'
        
        self.PlotAcryROC(classifier,test_y,predict,precision,recall,accuracy)

        
        FileMethod.SaveModelParam(filesAll,preproAll,ANNNET)
        self.train_flag = True
        DrawPic(self.canvas,self.fig_sub)
        saveModel_press(model,self.StateQueue)
        print"model train end"


def refreshDataParams(preproParam):
    ###data preprocessing   
    preproParam.setWave(int(gui.inputwlengths.get()),int(gui.inputwlengthe.get()))
    preproParam.setTransF(gui.inputtransform.get())
    preproParam.setLamStr(gui.inputLambda.get())

    wlengths = int(gui.inputwlengths.get())
    wlengthe = int(gui.inputwlengthe.get())
    preproParam.setWave(wlengths,wlengthe)

    preproParam.setDropLabel(int(gui.inputLabel.get()))
    preproParam.setDataRate(float(gui.inputRate.get())) 
    
def refreshModelParams(modelParam):
    ##model params refresh
    modelParam.setOutDimention(int(gui.DL_Model_outputdimention.get()))
    modelParam.setACTFUNC(gui.DL_ActFunc.get())
    modelParam.setDropoutRate(float(gui.DL_Model_droprate.get()))
    modelParam.setGNsigma(float(gui.DL_GNsigma.get()))
    modelParam.setLayersNum(int(gui.DL_Model_layers.get()))
    modelParam.setOPTFUNC(gui.DL_Optimize.get())   
    modelParam.setRegularFUNC(gui.DL_Regularization.get())
    modelParam.setRegularRate(float(gui.DL_Regularization_rate.get()))

def refreshTrainParams(trainParam):
    st1batch = int(gui.step1_Batchsize.get())
    st2batch = int(gui.step2_Batchsize.get())
    st3batch = int(gui.step3_Batchsize.get())
    trainParam.setDLBatchsize(st1batch,st2batch,st3batch)
    
    st1lrate = float(gui.step1_DL_Learning_rate.get())   
    st2lrate = float(gui.step2_DL_Learning_rate.get())
    st3lrate = float(gui.step3_DL_Learning_rate.get())
    trainParam.setDLLearningRate(st1lrate,st2lrate,st3lrate)
    
    st1iter = int(gui.step1_Iterations.get())
    st2iter = int(gui.step2_Iterations.get())
    st3iter = int(gui.step3_Iterations.get())
    trainParam.setDLIterations(st1iter,st2iter,st3iter)
    
def refreshParam(datapara,modelpara,trainpara):
    print "Refresh Params start..."
    ###data preprocessing   
    refreshDataParams(datapara)
    refreshModelParams(modelpara)
    refreshTrainParams(trainpara)
    print "Reresh Params done." 
        
def DrawPic(canvas,fig_sub):
    canvas.get_tk_widget().grid(row=1,column=0,columnspan=4)    ##第一行显示,跨3列显示
    fig_sub.set_title('Model Classifier')
    fig_sub.set_xmargin
    fig_sub.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    fig_sub.set_xlim([-0.05, 1.15])  
    fig_sub.set_ylim([-0.05, 1.15])
    
    fig_sub.set_xlabel('False Positive Rate')  
    fig_sub.set_ylabel('True Positive Rate')  
    fig_sub.set_title('Receiver operating characteristic example')  
    fig_sub.legend(loc="lower right")

    canvas.show()


if __name__ == '__main__':
    ##构造对象：      
    filesAll = ParamsFile.FileNameParams()
    preproAll = ParamsFile.PreProParams()
    dltrainAll = DLModel.DeepLearnTrainParams()
    
    boolAll = ParamsFile.BoolFuncParams()
    plotAll = ParamsFile.ROCParams()
    
    ##
    ANNNET = DLModel.ANNETModel()
    sess = tf.InteractiveSession()
    testData = ParamsFile.TestData()
    ROC = ParamsFile.ROCParams()
    Report = ParamsFile.Result()

    ##开辟一块画布     
    root=Tk()
    root.title("NIR ANALYSIS PLATFORM(DL)")

    gui = MainGui(root)   
                                                                            
    root.resizable(False,False)##禁止改变窗口大小  
    #启动事件循环
    root.mainloop()
    
    ##befor exit must stop the trainthread
    gui.stop_trainthread()
    root.quit()
