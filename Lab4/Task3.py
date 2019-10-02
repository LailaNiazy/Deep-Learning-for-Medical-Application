# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:32:19 2019

@author: looly
"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from Data_Augmentation import generator_with_weights
from Data_Loader import get_train_test_data
from Dice import dice_coef_loss, dice_coef
from plotter import plotter
from u_net import u_net
import numpy as np
from sklearn.model_selection import KFold


def get_posterior_fold(s,counter,train_index,test_index):
    
    y_pred=np.load('Lab4'+'posterior_unet/step'+str(s -1)+'.npy')
    y_pred_train = y_pred[(counter*len(train_index)):((counter+1)*(train_index))]
    y_pred_val = y_pred[((counter+1)*(len(train_index)+1)):((counter+1)*sum(len(train_index),len(test_index)))]
        
    return y_pred_train, y_pred_val

def task_3():
    
    #Model parameters
    base = 16
    image_size = 240
    img_ch = 2
    batch_size =8
    LR = 0.00001
    SDRate = 0.5
    batch_normalization = True
    spatial_dropout = True
    epochs = 150
    final_neurons= 1 #binary classification
    final_afun = "softmax" #activation function

    #Data loader parameters
    p = 0.2
    path = '/Lab1/Lab3/MRI/'
    fold1 = 'Image'
    fold2 = 'Mask'


    #Weighted Dice
    weight_maps_bool = False
    
    #Load the data
    images, masks, _ = get_train_test_data(fold1, fold2, path, p, image_size, image_size, weight_maps_bool)
    #building model
    model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout,final_neurons, final_afun)
    #Compile the model
    print('compile')
    model.compile(optimizer = Adam(lr=LR), loss = dice_coef_loss, metrics =[dice_coef, Recall(), Precision()] )
    
    #number of cycles
    n_splits = 3      
    T = 2 
    for s in range(T):
        
        #k-fold crossvalidation loop
        cvscores = []   
        counter = 0
        model_predictions = []
        cv = KFold(n_splits=n_splits, random_state=42, shuffle=False)
        
        for train_index, test_index in cv.split(images):
            
            print('cross validation fold{}'.format(counter))
            x_train, x_val, y_train, y_val = images[train_index], images[test_index], masks[train_index], masks[test_index]

            if s == 0:
                train_predictions = np.ones((len(train_index),240,240,1))*0.5
                val_predictions = np.ones((len(test_index),240,240,1))*0.5
                print('here')
                x_train=np.concatenate((x_train,train_predictions),axis=-1)
                x_val=np.concatenate((x_val,val_predictions),axis=-1)
                print('here')
            else:
                y_pred_train, y_pred_val = get_posterior_fold(s,counter,train_index,test_index)
                x_train=np.concatenate((x_train,y_pred_train),axis=-1)
                x_val=np.concatenate((x_val,y_pred_val),axis=-1) 
                
            #training the model
            print('fit')
            History = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 1,
                            validation_data = (x_val, y_val))
            print('predict')
            #predict and add val_predictions to training
            train_predictions = model.predict(x_train,batch_size=int(batch_size/2))
            model_predictions[(counter*len(x_train)):((counter+1)*len(x_train))] = train_predictions
            val_predictions=model.predict(x_val,batch_size=int(batch_size/2))
            model_predictions[((counter+1)*(len(x_train)+1)):((counter+1)* sum(len(x_train),len(x_val)))] = val_predictions
            print('plotting')
            counter=+1
         #   print("%s: %.2f%%" % (model.metrics_names[1], History.history["val_dice_coef"]*100))
            cvscores.append(History.history["val_dice_coef"][len(History.history["val_dice_coef"])-1] * 100)
            fig_loss, fig_dice = plotter(History)
            fig_loss.savefig('/Plots/Task3/Learning_curve_Task{}_fold{}_step{}.png'.format(3,counter,s))
            fig_dice.savefig('/Plots/Task3/Dice_Score_Curve_Task{}_fold{}_step{}.png'.format(3,counter,s))
        np.save('Lab4'+'posterior_unet/step'+str(s)+'.npy', model_predictions)
            
#    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    return History