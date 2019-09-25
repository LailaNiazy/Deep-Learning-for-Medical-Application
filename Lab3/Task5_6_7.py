# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:58:29 2019

@author: looly
"""

from tensorflow.keras.optimizers import Adam
from Dice import dice_coef, dice_coef_loss
from u_net import u_net
from Data_Loader import get_train_test_data
from Data_Augmentation import DataAugmentation
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.utils import to_categorical
import sys
import plotter

def main(argv):
    task = argv[0]
    #Model parameters
    base = 16
    image_size = 256
    img_ch = 1
    batch_size =8
    LR = 0.0001
    SDRate = 0.5
    batch_normalization = True
    spatial_dropout = True
    epochs = 150
    
    
    #Data loader parameters
    p = 0.2
    fold1 = 'Image'
    fold2 = 'Mask'
    
    #Data augmentation parameters
    rotation_range = 10
    width_shift = 0.1
    height_shift_range = 0.1,
    rescale = 1./255
    horizontal_flip = True
    
    
    if task == '5a':
        
        number_of_labels = 1
        path = '/Lab1/Lab3/CT/'
        train_img, train_mask, test_img, test_mask = get_train_test_data(fold1, fold2, path, p,image_size, image_size)
        model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout, number_of_labels )
        
        model.compile(optimizer = Adam(lr=LR), loss = dice_coef_loss, metrics =[dice_coef] )
        History = model.fit(train_img, train_mask, epochs = epochs, batch_size = batch_size, verbose = 1,
                            validation_data = (test_img,test_mask))
        
        plotter(History)
    
    elif task == '5b':
        
        number_of_labels = 1
        path = '/Lab1/Lab3/CT/'
        #Load the data
        train_img, train_mask, test_img, test_mask = get_train_test_data(fold1, fold2, path, p,image_size, image_size)
        
        #Data augmentation
        train_datagen, val_datagen = DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip)
        #Build the model
        model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout, number_of_labels)
        
        #Compile the model
        model.compile(optimizer = Adam(lr=LR), loss = dice_coef_loss, metrics =[dice_coef, Recall(), Precision()] )
        
        #Fit the data into the model
        History = model.fit_generator(train_datagen.flow(train_img, train_mask,batch_size = batch_size), validation_data = val_datagen.flow(test_img, test_mask), epochs = epochs, verbose = 1)        
        
        #Plot results
        plotter(History, recall = True, precision = True)
        
    elif task == '6':
        
        number_of_labels = 3
        path = '/Lab1/Lab3/CT/'
        #Load the data
        train_img, train_mask, test_img, test_mask = get_train_test_data(fold1, fold2, path, p,image_size, image_size)
        
        #To one-hot-encoding
        train_mask = to_categorical(train_mask, num_classes=3)
        test_mask = to_categorical(test_mask, num_classes=3)
        
        #Data augmentation
        train_datagen, val_datagen = DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip)
        
        #Build the multi-classification model
        model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout,number_of_labels)
        
        #Compile the model
        model.compile(optimizer = Adam(lr=LR), loss = dice_coef_loss, metrics =[dice_coef, Recall(), Precision()] )
        
        #Fit the data into the model
        History = model.fit_generator(train_datagen.flow(train_img, train_mask,batch_size = batch_size), validation_data = val_datagen.flow(test_img, test_mask), epochs = epochs, verbose = 1)          
        
        #Plot results
        plotter(History, recall = True, precision = True)
    
    elif task == '7':
    
        path = '/Lab1/Lab3/MRI/'
        number_of_labels = 1
    
        #Load the data
        train_img, train_mask, test_img, test_mask = get_train_test_data(fold1, fold2, path, p,image_size, image_size)
        
        #Data augmentation
        train_datagen, val_datagen = DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip)
        
        #Build the multi-classification model
        model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout,number_of_labels)
        
        #Compile the model
        model.compile(optimizer = Adam(lr=LR), loss = dice_coef_loss, metrics =[dice_coef, Recall(), Precision()] )
        
        #Fit the data into the model
        History = model.fit_generator(train_datagen.flow(train_img, train_mask,batch_size = batch_size), validation_data = val_datagen.flow(test_img, test_mask), epochs = epochs, verbose = 1)          
        
        #Plot results
        plotter(History, recall = True, precision = True)
    else: 
        print('Wrong task number')
        
if __name__ == "__main__": 
    #input in the console is the number of the task
    main(sys.argv[1:])  
    
