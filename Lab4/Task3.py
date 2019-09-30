# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:32:19 2019

@author: looly
"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from Data_Augmentation import DataAugmentation
from Data_Loader import get_train_test_data
from sklearn.model_selection import train_test_split
from Dice import weighted_dice_loss, weighted_dice_coef
from plotter import plotter
from u_net import u_net
import numpy as np
from sklearn.model_selection import KFold


def task3():
    
     #Model parameters
    base = 16
    image_size = 240
    img_ch = 1
    batch_size =8
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

    #Data augmentation parameters
    data_augmentation = True
    rotation_range = 10
    width_shift = 0.1
    height_shift_range = 0.1,
    rescale = 1./255
    horizontal_flip = True

    #K-fold cross validation
    weight_strength = 1
    weight_maps_bool = True
    T = 2 #number of cycles
    
    x_train,x_val,y_train,y_val = train_test_split(images,masks,test_size = p, shuffle = False)
    images, masks, weight_maps = get_train_test_data(fold1, fold2, path, p,image_size, image_size)
    #building model
    model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout,final_neurons, final_afun)
    
    for s in range(T):
        #train_test split
        
        weight_train = weight_map[0:x_train.shape[0]-1][:,:]
        weight_val = weight_map[x_train.shape[0]:][:,:]
        #Compile the model
        model.compile(optimizer = Adam(lr=LR), loss = weighted_loss(weight_maps, weight_strength), metrics =[dice_coef, Recall(), Precision()] )
        if data_augmentation:
            #Fit the data into the model
            train_generator = generator_with_weights(x_train, y_train, weight_train, Batch_size)
            History = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, 
                                                verbose=1, max_queue_size=1, validation_steps=len(x_val),
                                                validation_data=([x_val, weight_val], y_val), shuffle=True, class_weight='auto')
        else:
            History = model.fit([x_train, weight_train], y_train, epochs = epochs, batch_size = batch_size, verbose = 1,
                            validation_data = ([x_val, weight_val], y_val))
            
        
        val_predictions=model.predict(x_val,batch_size=int(batch_size/2))
        train_predictions=model.predict(x_train,batch_size=int(batch_size/2))
        
        model_predictions[(s*1):((s+1)*1)] = val_predictions
        
        #Ask, we need to change the channels to 2
    
        x_train,x_val,y_train,y_val = train_test_split(images,masks,test_size = p, shuffle = False)
        
        x_train=np.concatenate((x_train,train_predictions),axis=-1)
        
        x_val=np.concatenate((x_val,val_predictions),axis=-1)

        plotter(History)
    
    return History