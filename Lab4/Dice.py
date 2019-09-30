# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:34:54 2019

"""

##similarity metrics
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred):
#calculate the dice coefficient for two images
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    #calculate the dice loss using the dice coefficient
    return 1.-dice_coef(y_true, y_pred)
    
    
def weighted_dice_loss(weight_map, weight_strength):
    def weighted_dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        weight_f = K.flatten(weight_map)
        weight_f = weight_f * weight_strength +1
        wy_true_f = weight_f * y_true_f
        wy_pred_f = weight_f * y_pred_f
        return dice_coef(wy_true_f,wy_pred_f)
    return -weighted_dice_coef
