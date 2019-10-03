# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:34:03 2019

@author: looly
"""

##Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip):
    
    #Train data
    train_datagen = ImageDataGenerator(rotation_range = rotation_range, width_shift_range = width_shift, height_shift_range=height_shift_range,horizontal_flip = horizontal_flip, rescale = rescale)
   
    #Val data
    val_datagen = ImageDataGenerator(rescale = rescale)
 
    
    return train_datagen, val_datagen 