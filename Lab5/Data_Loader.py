# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:32:27 2019

@author: looly
"""

import tensorflow as tf


tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)

##image loader

import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split


def path_loader(fold1, fold2, data_path):
    #Creating data path
    image_data_path = os.path.join(data_path, fold1)   
    mask_data_path = os.path.join(data_path, fold2)
    images = []
    masks = []
    #Listing all file names in the path
    for root, dirs, files in os.walk(image_data_path):
        for name in files:
            images.append(os.path.join(image_data_path,name))
    for root2, dirs2, files2 in os.walk(mask_data_path):
        for name2 in files2:
            masks.append(os.path.join(mask_data_path,name2))
    return images, masks
    

def get_train_data_shuffled(images, masks, p):
    
    #shuffling the data and splitting it to test and train set according to the percentage p
    #the reason for zipping is to get the images and masks shuffle the same order
    c = list(zip(images, masks))

    shuffle(c)

    images, masks = zip(*c)
    #splitting the training and test data
    
    train_x, test_x, train_y, test_y = train_test_split(images,masks,test_size = p)

    return train_x, test_x, train_y, test_y 

# reading and resizing the training images with their corresponding labels
def data_loader(fold1, fold2, data_path, p,img_h, img_w):
    
    images, masks = path_loader(fold1, fold2, data_path)
    train_x, test_x, train_y, test_y = get_train_data_shuffled(images, masks, p)
    
    train_img = []
    train_mask = []
    test_img = []
    test_mask = []

    for i in range(len(train_x)):
        image_name = train_x[i]
        img = imread(image_name, as_grey=True)
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        train_img.append([np.array(img)]) 

        if i % 50 == 0:
             print('Reading: {0}/{1}  of train images'.format(i, len(train_x)))
                
    for j in range(len(train_y)):
        mask_name = train_y[j]
        mask = imread(mask_name, as_grey=True)
        mask = resize(mask, (img_h, img_w), anti_aliasing = True).astype('float32')
        train_mask.append([np.array(mask)])
        
    for i in range(len(test_x)):
        image_name = test_x[i]
        img = imread(image_name, as_grey=True)
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        test_img.append([np.array(img)]) 

        if i % 50 == 0:
             print('Reading: {0}/{1}  of test images'.format(i, len(test_x)))
                
    for j in range(len(test_y)):
        mask_name = test_y[j]
        mask = imread(mask_name, as_grey=True)
        mask = resize(mask, (img_h, img_w), anti_aliasing = True).astype('float32')
        test_mask.append([np.array(mask)])        
        
 
    return train_img, train_mask, test_img, test_mask

# Instantiating images and labels for the model.
def get_train_test_data(fold1, fold2, data_path, p,img_h, img_w):
    
    train_img, train_mask, test_img, test_mask = data_loader(fold1, fold2, data_path, p,img_h, img_w)

    Train_Img = np.zeros((len(train_img), img_h, img_w), dtype = np.float32)
    Test_Img = np.zeros((len(test_img), img_h, img_w), dtype = np.float32)

    Train_Label = np.zeros((len(train_mask),img_h, img_w), dtype = np.float32)
    Test_Label = np.zeros((len(test_mask),img_h, img_w), dtype = np.float32)

    for i in range(len(train_img)):
        Train_Img[i] = train_img[i][0]
        Train_Label[i] = train_mask[i][0]

    Train_Img = np.expand_dims(Train_Img, axis = 3)  
    Train_Label = np.expand_dims(Train_Label, axis = 3) 

    for j in range(len(test_img)):
        Test_Img[j] = test_img[j][0]
        Test_Label[j] = test_mask[j][0]

    Test_Img = np.expand_dims(Test_Img, axis = 3)
    Test_Label = np.expand_dims(Test_Label, axis = 3)


    return Train_Img, Train_Label, Test_Img, Test_Label
