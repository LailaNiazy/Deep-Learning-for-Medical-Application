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
    image_size = 240
    img_ch = 1
    batch_size =8
    LR = 0.0001
    SDRate = 0.5
    batch_normalization = True
    spatial_dropout = True
    epochs = 10
    final_neurons= 1 #binary classification
    final_afun = "sigmoid" #activation function

    #Data loader parameters
    p = 0.2
    path = '/Lab1/Lab3/MRI/'
    fold1 = 'Image'
    fold2 = 'Mask'
    weight_maps_flag = False

    #Data augmentation parameters
    rotation_range = 10
    width_shift = 0.1
    height_shift_range = 0.1,
    rescale = 1./255
    horizontal_flip = True

    #K-fold cross validation
    n_folds = 3
    #Load the data
    print('getting data')
    images, masks, weight_maps = get_train_test_data(fold1, fold2, path, p,image_size, image_size,weight_maps_flag)

    #Data augmentation
    print('Data augmentation')
    train_datagen, val_datagen = DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip)


    #Build the model
    print('building the model')
    model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout,final_neurons, final_afun)
    
    #Compile the model
    model.compile(optimizer = Adam(lr=LR), loss = dice_coef_loss, metrics =[dice_coef, Recall(), Precision()] )

    cvscores = []
    #k-fold crossvalidation loop
    cv = KFold(n_splits=3, random_state=42, shuffle=False)
    counter = 1
    #for _ in range(n_folds):
    for train_index, test_index in cv.split(images):
        #train_test split
        print('cross validation fold{}'.format(counter))
        #x_train,x_val,y_train,y_val = train_test_split(images,masks,test_size = p)
        print(train_index)
        x_train, x_val, y_train, y_val = images[train_index], images[test_index], masks[train_index], masks[test_index]
        
        #Fit the data into the model
        History = model.fit_generator(train_datagen.flow(x_train, y_train,batch_size = batch_size), validation_data = val_datagen.flow(x_val, y_val), epochs = epochs, verbose = 1)

        print("%s: %.2f%%" % (model.metrics_names[1], History.history["val_dice_coef"]*100))

        cvscores.append(History.history["val_dice_coef"][len(History.history["val_dice_coef"])-1] * 100)

        fig_loss, fig_dice = plotter(History)

        fig_loss.savefig('/Task_1/Learning_curve_{}_fold{}.png'.format(1,counter))
        fig_dice.savefig('/Task_1/Dice_Score_Curve_{}_fold{}.png'.format(1,counter))
        
        counter=counter+1
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    return History