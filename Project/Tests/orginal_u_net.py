import matplotlib.pyplot as plt
from Utils.Images import get_train_test_data
from tensorflow.keras.optimizers import Adam
from Utils.plotter import plotter
from Utils.losses import jaccard_loss
from Utils.u_net import u_net
from Utils.Data_Augmentation import DataAugmentation
from sklearn.model_selection import KFold
import sys
import numpy as np
from sklearn.utils import class_weight
import json

def orginal_u_net():
    
    #########Initialize the data for the images#############
    img_ch = 1
    number_of_labels = 1
    fold1 = '01' #folder with training data
    fold2 = '01_GT/TRA' #folder with ground truth
    data_path = 'PhC-C2DH-U373'
    
    #########Initialize the data for the augmentation#############
    rotation_range = 10
    width_shift = 0.1
    height_shift = 0.1
    rescale = 0.2
    horizontal_flip = True

    #########Import the different configurations for model#############
    with open('intialize_unet.json', 'r') as f:
        distros_dict = json.load(f)
        
    for distro in distros_dict: 

        #########Data Augmentation#############
        train_datagen, val_datagen = DataAugmentation(rotation_range,width_shift,height_shift,rescale,horizontal_flip)
        #########Get the images and their masks (labels)#############
        images, masks, _ = get_train_test_data(fold1, fold2, data_path, distro['img_h'], distro['img_w'])
        #########Initialize the model and compile it #############
        model = u_net(distro['base'],distro['img_h'], distro['img_w'], img_ch, distro['batch_normalization'], distro['SDRate'], distro['spatial_dropout'], number_of_labels)
        model.compile(optimizer = Adam(lr=distro['LR']), loss = "binary_crossentropy", metrics =[jaccard_loss])

        #########k-fold crossvalidation loop#############  
        counter = 0
        cv = KFold(n_splits=3, random_state=42, shuffle=False)
        for train_index, test_index in cv.split(images):

                #splitting the data into training and test sets
                print('cross validation fold{}'.format(counter))
                x_train, x_val, y_train, y_val = images[train_index], images[test_index], masks[train_index], masks[test_index]

                # Calculate the weights for each class so that we can balance the data
                y = y_train.reshape(y_train.shape[0],y_train.shape[1]*y_train.shape[2]).shape
                weights = class_weight.compute_sample_weight('balanced',y)

                # Add the class weights to the training and fit the data into the model  
                if distro['data_augmentation']:
                    History = model.fit_generator(train_datagen.flow(x_train, y_train,batch_size = distro['batch_size']), validation_data = val_datagen.flow(x_val,y_val), epochs = distro['epochs'], verbose = 1, class_weight=weights)   
                else:
                    History = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=weights,verbose = 1 validation_data =(x_val,y_val))

                #plotting
                fig_loss, fig_acc = plotter(History)
                fig_loss.savefig('Plots/Learning_curve_Task{}_fold{}.png'.format(1,counter))
                fig_acc.savefig('Plots/Jaccard_Loss_Curve_Task{}_fold{}.png'.format(1,counter))
                #saving the model to the folder Models
                model.save('Models/U-Net_with_DA.h5')
                counter=counter+1
             
