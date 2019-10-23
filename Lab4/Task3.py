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
    
    y_pred=np.load('posterior_unet/step'+str(s -1)+'.npy')
    print(len(train_index))
    print(len(test_index))
    i = counter*sum([len(train_index),len(test_index)])
    j = i + len(train_index)
    k = j + len(test_index)
    y_pred_train = y_pred[i:j]
    y_pred_val = y_pred[j:k]
        
    return y_pred_train, y_pred_val

def task_3():
    
    #Model parameters
    base = 16
    image_size = 128
    img_ch = 2 #here we have two channels
    batch_size =2
    LR = 0.00001
    SDRate = 0.5
    batch_normalization = True
    spatial_dropout = True
    epochs = 150
    final_neurons= 1 #binary classification
    final_afun = "sigmoid" #activation function

    #Data loader parameters
    p = 0.2
    path = '/Lab1/Lab3/MRI/'
    fold1 = 'Image'
    fold2 = 'Mask'


    #Weighted Dice
    weight_maps_bool = False
    
    #Load the data
    images, masks, _ = get_train_test_data(fold1, fold2, path, p, image_size, image_size)
    #building model
    model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout,final_neurons, final_afun)
    #Compile the model
    print('compile')
    model.compile(optimizer = Adam(lr=LR), loss = dice_coef_loss, metrics =[dice_coef, Recall(), Precision()] )
    
    #number of cycles
    n_splits = 3      
    T = 3 
    for s in range(T):
        print(s)
        #k-fold crossvalidation loop
        cvscores = []   
        counter = 0
        model_predictions = np.zeros([images.shape[0]*3,images.shape[1],images.shape[2],images.shape[3]])
        print(model_predictions.shape)
        cv = KFold(n_splits=n_splits, random_state=42, shuffle=False)
        
        for train_index, test_index in cv.split(images):
            print(counter)
            print('cross validation fold{}'.format(counter))
            x_train, x_val, y_train, y_val = images[train_index], images[test_index], masks[train_index], masks[test_index]
            
            if s == 0:
                train_predictions = np.ones((len(train_index),image_size,image_size,1))*0.5
                print(train_predictions.shape)
                val_predictions = np.ones((len(test_index),image_size,image_size,1))*0.5
                x_train=np.concatenate((x_train,train_predictions),axis=-1)  
                x_val=np.concatenate((x_val,val_predictions),axis=-1)
            else:
                print("here")
                y_pred_train, y_pred_val = get_posterior_fold(s,counter,train_index,test_index)
                x_train=np.concatenate((x_train,y_pred_train),axis=-1)
                print(x_train.shape)
                x_val=np.concatenate((x_val,y_pred_val),axis=-1) 
                
            #training the model
            print('fit')
            History = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_val, y_val))
            print('predict')
            #predict and add val_predictions to training
            train_predictions = model.predict(x_train,batch_size=int(batch_size/2))
            print(train_predictions.shape)
            #print((counter*len(x_train))
           # print((counter+1)*(len(x_train)))
            i = counter*sum([len(x_train),len(x_val)])
            j = i + len(x_train)
            k = j + len(x_val)
            print(i,j,k)
            model_predictions[i:j] = train_predictions
            val_predictions=model.predict(x_val,batch_size=int(batch_size/2))
          #  print((counter+1)*(len(x_train)))
          #  print((counter+1)* sum([len(x_train),len(x_val)]))
            print(val_predictions.shape)
            model_predictions[j:k] = val_predictions
            print('plotting')
            
            cvscores.append(History.history["val_dice_coef"][len(History.history["val_dice_coef"])-1] * 100)
            fig_loss, fig_dice = plotter(History)
            fig_loss.savefig('Plots/Task3/Learning_curve_Task{}_fold{}_step{}.png'.format(3,counter,s))
            fig_dice.savefig('Plots/Task3/Dice_Score_Curve_Task{}_fold{}_step{}.png'.format(3,counter,s))
            counter+=1
        np.save('posterior_unet/step'+str(s)+'.npy', model_predictions)
            
#    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    return History
