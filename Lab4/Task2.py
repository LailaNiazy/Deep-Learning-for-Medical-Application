# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:21:50 2019

"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from Data_Augmentation import DataAugmentation
from Data_Loader import get_train_test_data
from Dice import weighted_dice_coef, weighted_dice_loss
from plotter import plotter
from u_net import u_net
from Data_Augmentation import generator_with_weights
import numpy as np
from sklearn.model_selection import KFold



def task_2():

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
    #Load the data
    images, masks, weight_maps = get_train_test_data(fold1, fold2, path, p,image_size, image_size,weight_maps_bool)

    #Data augmentation
    train_datagen, val_datagen = DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip)


    #Build the model
    model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout,final_neurons, final_afun)
    #Compile the model
    model.compile(optimizer = Adam(lr=LR), loss = weighted_dice_loss(weight_maps, weight_strength), metrics =[weighted_dice_coef, Recall(), Precision()] )

    #k-fold crossvalidation loop
    cvscores = []
    #k-fold crossvalidation loop
    cv = KFold(n_splits=3, random_state=42, shuffle=False)
    
    counter = 1
    for train_index, test_index in cv.split(images):
        #train_test split
        print('cross validation fold{}'.format(counter))
        #x_train,x_val,y_train,y_val = train_test_split(images,masks,test_size = p)
        x_train, x_val, y_train, y_val = images[train_index], images[test_index], masks[train_index], masks[test_index]
        weight_train = weight_maps[train_index]
        weight_val = weight_maps[test_index]
        if data_augmentation:
            #Fit the data into the model
            train_generator = generator_with_weights(x_train, y_train, weight_train, batch_size)
           # val_generator = generator_with_weights(x_val, y_val, weight_val, batch_size)
            History = model.fit_generator(train_generator, epochs=epochs, 
                                                verbose=1, max_queue_size=1, validation_steps=len(x_val),
                                                validation_data=([x_val, weight_val], y_val), shuffle=True, class_weight='auto')
        else:
            History = model.fit([x_train, weight_train], y_train, epochs = epochs, batch_size = batch_size, verbose = 1,
                            validation_data = ([x_val, weight_val], y_val))


        print("%s: %.2f%%" % (model.metrics_names[1], History.history["val_dice_coef"]*100))
        cvscores.append(History.history["val_dice_coef"][len(History.history["val_dice_coef"])-1] * 100)
        fig_loss, fig_dice = plotter(History)

        fig_loss.savefig('/Task_1/Learning_curve_{}_fold{}.png'.format(2,counter))
        fig_dice.savefig('/Task_1/Dice_Score_Curve_{}_fold{}.png'.format(2,counter))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    return History
