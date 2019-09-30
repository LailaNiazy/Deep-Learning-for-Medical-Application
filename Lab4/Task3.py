# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:32:19 2019

@author: looly
"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from Data_Augmentation import generator_with_weights
from Data_Loader import get_train_test_data
from Dice import weighted_dice_loss, weighted_dice_coef
from plotter import plotter
from u_net import u_net
import numpy as np
from sklearn.model_selection import KFold


def task3():
    
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
    final_afun = "sigmoid" #activation function

    #Data loader parameters
    p = 0.2
    path = '/Lab1/Lab3/MRI/'
    fold1 = 'Image'
    fold2 = 'Mask'


    #Weighted Dice
    weight_strength = 1
    weight_maps_bool = True
    
    
    
    #Load the data
    images, masks, weight_maps = get_train_test_data(fold1, fold2, path, p,image_size, image_size,weight_maps_bool)
    #building model
    model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout,final_neurons, final_afun)
    #Compile the model
    model.compile(optimizer = Adam(lr=LR), loss = weighted_dice_loss(weight_maps, weight_strength), metrics =[weighted_dice_coef, Recall(), Precision()] )
          
    T = 2 #number of cycles
    counter = 1
    for s in range(T):
        cvscores = []
        #k-fold crossvalidation loop
        cv = KFold(n_splits=3, random_state=42, shuffle=False)
        for train_index, test_index in cv.split(images):
            
            print('cross validation fold{}'.format(counter))
            #x_train,x_val,y_train,y_val = train_test_split(images,masks,test_size = p)
            x_train, x_val, y_train, y_val = images[train_index], images[test_index], masks[train_index], masks[test_index]
            weight_train = weight_maps[train_index]
            weight_val = weight_maps[test_index]
            
           
            
            if s == 0:
                train_predictions = np.ones(240,240,1)*0.5
                val_predictions = np.ones(240,240,1)*0.5
                x_train=np.concatenate((x_train,train_predictions),axis=-1)
                x_val=np.concatenate((x_val,val_predictions),axis=-1)
            else:
                y_pred=np.load('Lab4'+'posterior_unet/step'+str(step -1)+'.npy')
                x_train=np.concatenate((x_train,train_predictions),axis=-1)
                x_val=np.concatenate((x_val,val_predictions),axis=-1)    
                #Fit the data into the model
            train_generator = generator_with_weights(x_train, y_train, weight_train, batch_size)
            History = model.fit_generator(train_generator, epochs=epochs, 
                                verbose=1, max_queue_size=1, validation_steps=len(x_val),
                                validation_data=([x_val, weight_val], y_val), shuffle=True, class_weight='auto')
   
            
            val_predictions=model.predict(x_val,batch_size=int(batch_size/2))
            train_predictions=model.predict(x_train,batch_size=int(batch_size/2))
            
            model_predictions[(s*1):((s+1)*1)] = val_predictions            
            np.save('Lab4'+'posterior_unet/step'+str(counter)+'.npy', model_predictions)
            #Ask, we need to change the channels to 2
            

            counter=+1
            print("%s: %.2f%%" % (model.metrics_names[1], History.history["val_dice_coef"]*100))
            cvscores.append(History.history["val_dice_coef"][len(History.history["val_dice_coef"])-1] * 100)
            fig_loss, fig_dice = plotter(History)
    
            fig_loss.savefig('/Task_1/Learning_curve_{}_fold{}.png'.format(2,counter))
            fig_dice.savefig('/Task_1/Dice_Score_Curve_{}_fold{}.png'.format(2,counter))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    
    return History