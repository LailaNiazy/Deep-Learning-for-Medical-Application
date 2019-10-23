# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:21:50 2019

"""

from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.metrics import Recall, Precision
import numpy as np
from Data_Augmentation import DataAugmentation
from Data_Loader import get_train_test_data
from Dice import dice_coef_loss, dice_coef
from plotter import plotter
from u_net import u_net
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def task_1():


    #Model parameters
    base = 16
    image_size = 128
    img_ch = 1
    batch_size =2
    LR = 0.0001
    SDRate = 0.5
    batch_normalization = True
    spatial_dropout = True
    epochs = 150
    final_neurons= 1 #binary classification
    final_afun = "sigmoid" #activation function
    weighted = False #paramter so we can differ between two outputs in case of a weighted dice fcn and a normal dice fcn
    #Data loader parameters
    p = 0.2
    path = '/Lab1/Lab3/MRI/'
    fold1 = 'Image'
    fold2 = 'Mask'

    #Data augmentation parameters
    data_augmenetation = False
    rotation_range = 10
    width_shift = 0.1
    height_shift_range = 0.1,
    rescale = 1./255
    horizontal_flip = True


    #Load the data
    images, masks, _ = get_train_test_data(fold1, fold2, path, p,image_size, image_size)

    #Data augmentation
    train_datagen, val_datagen = DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip)


     #k-fold crossvalidation loop
    #cvscores = []
    cv = KFold(n_splits=3, shuffle=False)
    counter = 1
    
    for train_index, test_index in cv.split(images):
        #train_test split
        print(counter)
        x_train, x_val, y_train, y_val = images[train_index], images[test_index], masks[train_index], masks[test_index]
        
        #Build the model
        model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout,final_neurons, final_afun,weighted)

        #Compile the model
        model.compile(optimizer = Adam(lr=LR), loss = [dice_coef_loss], metrics =[dice_coef, Recall(), Precision()] )
        
        #Fit the data into the model
        if data_augmenetation:
            History = model.fit_generator(train_datagen.flow(x_train, y_train,batch_size = batch_size), validation_data = val_datagen.flow(x_val, y_val), epochs = epochs, verbose = 1)
        else:
            History = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 1, validation_data = (x_val, y_val))
        
        #plotting and saving the plots
        fig_loss, fig_dice = plotter(History)
        fig_loss.savefig('Plots/Task1/Learning_curve_Task{}_fold{}_without_DA.png'.format(1,counter))
        fig_dice.savefig('Plots/Task1/Dice_Score_Curve_Task{}_fold{}_without_DA.png'.format(1,counter))
        counter=counter+1
    return History
