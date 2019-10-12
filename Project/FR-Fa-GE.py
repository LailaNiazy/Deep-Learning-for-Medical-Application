import matplotlib.pyplot as plt
from Images import get_train_test_data
from tensorflow.keras.optimizers import Adam
from plotter import plotter
from metrics import jaccard_distance
from losses import weighted_bce_loss
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
    
    
    train_img, train_mask, weight_maps = get_train_test_data(fold1, fold2, data_path, img_h, img_w)
    model = u_net(base, img_w, img_h, img_ch, batch_normalization, SDRate, spatial_dropout, final_neurons)
    model.compile(optimizer = Adam(lr=LR), loss = weighted_bce_loss(weight_maps, weight_strength), metrics =[jaccard_distance])
    cvscores = []
    cv = KFold(n_splits=3, random_state=42, shuffle=False)
    
    counter = 1
    for train_index, test_index in cv.split(train_img):
    
    #Fit the data into the model
        x_train, x_val, y_train, y_val = train_img[train_index], train_img[test_index], train_mask[train_index], train_mask[test_index]
        weight_train = weight_maps[train_index]
        weight_val = weight_maps[test_index]

        #Fit the data into the model
        train_generator = generator_with_weights(x_train, y_train, weight_train, batch_size)
        val_generator = generator_with_weights(x_val, y_val, weight_val, batch_size)
        History = model.fit_generator(train_generator, epochs=epochs, verbose=1, max_queue_size=1, validation_steps=len(x_val), validation_data=([x_val, weight_val], y_val), shuffle=False, class_weight='auto', steps_per_epoch = math.ceil(len(x_train) / batch_size))
        #Plots per fold
        fig_loss, fig_dice = plotter(History)
        fig_loss.savefig('/Plots/Learning_Curve_FR-Fa-GE{}_fold{}.png'.format(2,counter))
        fig_dice.savefig('/Plots/Dice_Score_Curve_FR-Fa-GE{}_fold{}.png'.format(2,counter))
        counter += 1

if __name__ == "__main__":
    #input in the console is the number of the task
    main()