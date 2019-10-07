# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:36:08 2019

@author: looly
"""

from tensorflow.keras.optimizers import Adam
from plotter import plotter
from u_net import u_net
from Dice import dice_coef, dice_coef_loss
from Data_Loader_Task3 import get_train_test_data
from Data_Augmentation import DataAugmentation
import sys

def task_3():
    
    image_size = 256 #both width and height of image are the same
    img_ch = 1
    batch_size = 8
    LR = 0.0001
    SDRate = 0.5
    spatial_dropout = True
    epochs = 150
    base  = 32
    batch_normalization = True
    p = 0.2 #percentage of training and test data
    path = '/Lab1/Lab3/X_ray/'
    fold1 = 'Image'
    fold2 = 'Mask'
    number_of_labels = 1
    
    
    #Data Augmentation
    rotation_range = 10
    width_shift = 0.1
    height_shift = 0.1
    rescale = 0.2
    horizontal_flip = True
       
   # train_datagen, val_datagen = DataAugmentation(rotation_range,width_shift,height_shift,rescale,horizontal_flip)
    
    train_img, train_mask, test_img, test_mask = get_train_test_data(fold1, fold2, path, p,image_size, image_size)
    model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout, number_of_labels)
    model.compile(optimizer = Adam(lr=LR), loss = dice_coef_loss, metrics =[dice_coef])
    #Fit the data into the model
    History = model.fit(train_img, train_mask, epochs = epochs, batch_size = batch_size, verbose = 1,
                            validation_data = (test_img,test_mask))
 #   History = model.fit_generator(train_datagen.flow(train_img, train_mask,batch_size = batch_size), validation_data = val_datagen.flow(test_img, test_mask), epochs = epochs, verbose = 1)       
    plotter(History, '3')
    
    
 