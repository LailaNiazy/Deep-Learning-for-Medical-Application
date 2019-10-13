import matplotlib.pyplot as plt
from Images import get_train_test_data
from tensorflow.keras.optimizers import Adam
from plotter import plotter
from losses import *
from u_net import u_net
from Data_Augmentation import*
from sklearn.model_selection import KFold
import sys
import math

def main():
    
    #Data parameters
    img_h = 128
    img_w = 128
    img_ch = 1
    fold1 = '01'
    fold2 = '01_GT/TRA'
    data_path = 'PhC-C2DH-U373'
    
    #Network parameters
    base = 16
    batch_size = 1
    LR = 0.00001
    SDRate = 0.5
    batch_normalization = True
    spatial_dropout = True
    epochs = 150
    final_neurons= 1 #binary classification
    final_afun = "sigmoid" #activation function
    weight_strength = 1.
    
    #Get the images, masks and weight maps
    train_img, train_mask, weight_maps = get_train_test_data(fold1, fold2, data_path, img_h, img_w)
    
    #Compile the u-net model with the previously stated parameters
    model = u_net(base, img_w, img_h, img_ch, batch_normalization, SDRate, spatial_dropout, final_neurons)
    
    #k-fold crossvalidation loop
    cv = KFold(n_splits=3, random_state=42, shuffle=False)
    
    #fold counter
    counter = 1
    for train_index, test_index in cv.split(train_img):
  
        #divide the training and validation data
        x_train, x_val, y_train, y_val = train_img[train_index], train_img[test_index], train_mask[train_index], train_mask[test_index]
        
        #divide the weight maps data into train and test following the same order as the data
        weight_train = weight_maps[train_index]
        weight_val = weight_maps[test_index]
        
        #Compile the model with weighted cross-entropy loss and metric as jaccard_distance
        model.compile(optimizer = Adam(lr=LR), loss = weighted_bce_loss(weight_maps, weight_strength), metrics =[jaccard_loss])

        ##Fit the data into the model
        #Create the train generator for data augmentation
        train_generator = generator_with_weights(x_train, y_train, weight_train, batch_size)
        
        #Create the validation generator
        val_generator = generator_with_weights(x_val, y_val, weight_val, batch_size)
        
        #Fit the data into the model
        History = model.fit_generator(train_generator, epochs=epochs, verbose=1, max_queue_size=1, validation_steps=len(x_val), validation_data=([x_val, weight_val], y_val), shuffle=False, class_weight='auto', steps_per_epoch = math.ceil(len(x_train) / batch_size))
         
        #Save model
        model.save('Models/FR-Fa-GE/FR-Fa-GE_model_fold{}.h5'.format(counter))
        # Save the weights
        model.save_weights('Models/FR-Fa-GE/FR-Fa-GE_weights_fold{}.h5'.format(counter))

        # Save the model architecture
        with open('Models/FR-Fa-GE/FR-Fa-GE_architecture_fold{}.json'.format(counter), 'w') as f:
            f.write(model.to_json())
        
        #Plots per fold
        fig_loss, fig_dice = plotter(History)
        fig_loss.savefig('Plots/FR-Fa-GE/Learning_Curve_FR-Fa-GE_fold{}.png'.format(counter))
        fig_dice.savefig('Plots/FR-Fa-GE/Jaccard_Score_Curve_FR-Fa-GE_fold{}.png'.format(counter))
        counter += 1
        

if __name__ == "__main__":
    
    main()