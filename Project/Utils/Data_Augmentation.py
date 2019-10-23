# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:34:03 2019

"""

##Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip):
    
    #Train data
    train_datagen = ImageDataGenerator(rotation_range = rotation_range, width_shift_range = width_shift, height_shift_range=height_shift_range,
                                       horizontal_flip = horizontal_flip, rescale = rescale)
   
    #Val data
    val_datagen = ImageDataGenerator(rescale = rescale)
 
    
    
    return train_datagen, val_datagen 

def combine_generator(gen1, gen2, gen3):
     while True:
        x = gen1.next()
        y = gen2.next()
        w = gen3.next()
        yield([x, w], y)
        
def generator_with_weights(x_train, y_train, weights_train, batch_size):
    
    background_value = x_train.min()
    data_gen_args = dict(rotation_range=10., width_shift_range=0.1, height_shift_range=0.1, 
                         cval=background_value, zoom_range=0.2, horizontal_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    weights_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow(x_train, shuffle=False, batch_size=batch_size, seed=1)
    mask_generator = mask_datagen.flow(y_train, shuffle=False, batch_size=batch_size, seed=1)
    weight_generator = weights_datagen.flow(weights_train, shuffle=False, batch_size=batch_size, seed=1)
    train_generator = combine_generator(image_generator, mask_generator, weight_generator)
    
    return train_generator