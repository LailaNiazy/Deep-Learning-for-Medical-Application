import matplotlib.pyplot as plt
from Images import get_train_test_data
from tensorflow.keras.optimizers import Adam
from plotter import plotter
from metrics import jaccard, jaccard_distance
from u_net import u_net
from Data_Augmentation import DataAugmentation
from sklearn.model_selection import KFold
import sys
import numpy as np
from sklearn.utils import class_weight


def main():
    
    img_h = 240
    img_w = 240
    img_ch = 1
    number_of_labels = 1
    LR = 0.00001
    fold1 = '01'
    fold2 = '01_GT/TRA'
    data_path = 'PhC-C2DH-U373'
    base  = 32
    batch_size = 8
    epochs = 2
    spatial_dropout = True
    SDRate = 0.5
    
    batch_normalization = True
    rotation_range = 10
    width_shift = 0.1
    height_shift = 0.1
    rescale = 0.2
    horizontal_flip = True
    
    train_datagen, val_datagen = DataAugmentation(rotation_range,width_shift,height_shift,rescale,horizontal_flip)
    images, masks, _ = get_train_test_data(fold1, fold2, data_path, img_h, img_w)
    model = u_net(base,img_h, img_w, img_ch, batch_normalization, SDRate, spatial_dropout, number_of_labels)
    model.compile(optimizer = Adam(lr=LR), loss = "binary_crossentropy", metrics =[jaccard_distance])
    
    #k-fold crossvalidation loop
    cvscores = []   
    counter = 0
    cv = KFold(n_splits=3, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(images):
            
            print('cross validation fold{}'.format(counter))
            x_train, x_val, y_train, y_val = images[train_index], images[test_index], masks[train_index], masks[test_index]
            
            # Calculate the weights for each class so that we can balance the data
         #   weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
            # Add the class weights to the training and Fit the data into the model                                       
         #   History = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=weights,verbose = 1 validation_data =(x_val,y_val))

            #Fit the data into the model
            History = model.fit_generator(train_datagen.flow(x_train, y_train,batch_size = batch_size), validation_data = val_datagen.flow(x_val,y_val), epochs = epochs, verbose = 1)
                                          #class_weight=weights)      
            
            #plot
            fig_loss, fig_acc = plotter(History)
            fig_loss.savefig('Plots/Learning_curve_Task{}_fold{}.png'.format(1,counter))
            fig_dice.savefig('Plots/Accuracy_Curve_Task{}_fold{}.png'.format(1,counter))
            counter=counter+1

if __name__ == "__main__":
    #input in the console is the number of the task
    main()