# -*- coding: utf-8 -*-
from tensorflow.keras.optimizers import Adam
from LSTM_model2 import model_LSTM2
from plotter2 import plotter2
from Data_Loader_Task2 import data_loader_task2
from MyBatchGenerator import *
from tensorflow.keras.preprocessing.sequence import pad_sequences



def task_2():
    #Initialize data loading parameters
    dataPath = '/Lab1/Lab5/HCP_lab/'
    train_subjects_list = ['599469','599671','613538']
    val_subjects_list = ['601127']
    bundles_list = ['CST_left', 'CST_right']
    n_tracts_per_bundle = 20
    
    #load the training and test data
    X_train, y_train, X_val, y_val = data_loader_task2(dataPath, train_subjects_list,val_subjects_list, bundles_list, n_tracts_per_bundle)
    

    #initialize the model paramters
    batch_size = 1
    units = 5
    drop_rate = 0.2
    LR = 0.001
    n_epochs = 50
    input_dimension = (None,3)
    masking = False
    
    #generate the model and compile it
    model = model_LSTM2(drop_rate, units,input_dimension, masking)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LR), metrics=['binary_accuracy'])
    model.summary()

    #train the model 
    History = model.fit_generator(MyBatchGenerator(X_train, y_train, batch_size=batch_size), epochs=n_epochs, validation_data=MyBatchGenerator(X_val, y_val, batch_size=batch_size), validation_steps=len(X_val))
    
    
    #plot the loss, mae curve and stock_price curve
    plotter2(History,"2")
    
    return History

#Bonus task 1

def task_2_1():
    #Initialize data loading parameters
    dataPath = '/Lab1/Lab5/HCP_lab/'
    train_subjects_list = ['599469','599671','613538']
    val_subjects_list = ['601127']
    bundles_list = ['CST_left', 'CST_right']
    n_tracts_per_bundle = 20
    
    #load the training and test data
    X_train, y_train, X_val, y_val = data_loader_task2(dataPath, train_subjects_list,val_subjects_list, bundles_list, n_tracts_per_bundle)
    
    #initialize the model paramters
    batch_size = 2
    units = 10
    drop_rate = 0.2
    LR = 0.001
    n_epochs = 50
    input_dimension = (None,3)
    masking = True
    
    #generate the model and compile it
    model = model_LSTM2(drop_rate, units,input_dimension, masking)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LR), metrics=['binary_accuracy'])
    model.summary()
    
    #pad the data
    X_train = pad_sequences(X_train,padding='post')
    X_val = pad_sequences(X_val,padding='post')

    #train the model 
    History = model.fit_generator(MyBatchGenerator(X_train, y_train, batch_size=batch_size), epochs=n_epochs, validation_data=MyBatchGenerator(X_val, y_val, batch_size=batch_size), validation_steps=len(X_val))
    
    
    #plot the loss, mae curve and stock_price curve
    plotter2(History,"2_1")
    
    return History

def task_2_2():
    #Initialize data loading parameters
    dataPath = '/Lab1/Lab5/HCP_lab/'
    train_subjects_list = ['599469','599671','613538']
    val_subjects_list = ['601127']
    bundles_list = ['ST_FO_left', 'ST_FO_right']
    n_tracts_per_bundle = 20
    
    #load the training and test data
    X_train, y_train, X_val, y_val = data_loader_task2(dataPath, train_subjects_list,val_subjects_list, bundles_list, n_tracts_per_bundle)
    
    #initialize the model paramters
    batch_size = 2
    units = 10
    drop_rate = 0.2
    LR = 0.001
    n_epochs = 50
    input_dimension = (None,3)
    masking = True
    
    #generate the model and compile it
    model = model_LSTM2(drop_rate, units,input_dimension, masking)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LR), metrics=['binary_accuracy'])
    model.summary()
    
    #pad the data
    X_train = pad_sequences(X_train,padding='post')
    X_val = pad_sequences(X_val,padding='post')

    #train the model 
    History = model.fit_generator(MyBatchGenerator(X_train, y_train, batch_size=batch_size), epochs=n_epochs, validation_data=MyBatchGenerator(X_val, y_val, batch_size=batch_size), validation_steps=len(X_val))
    
    
    #plot the loss, mae curve and stock_price curve
    plotter2(History,"2_2")
    
    return History

