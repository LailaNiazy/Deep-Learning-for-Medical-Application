# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:36:08 2019

@author: looly
"""
from tensorflow.keras.optimizers import Adam
from LSTM_model import model_LSTM
from plotter import plotter
from Data_Loader_Task1 import data_loader


def task_1():
    
    #initialize the model paramters
    batch_size = 16
    units = 40
    drop_rate = 0.2
    LR = 0.001
    epochs = 100
    input_dimension = 1
    #load the training and test data
    X_train, y_train, X_val, y_val = data_loader()

    #generate the model and compile it
    model = model_LSTM(drop_rate, units, X_train.shape[1])
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=LR), metrics=['mean_absolute_error'])

    #train the model 
    History = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data = (X_val, y_val), verbose=2)
    
    #predict the prices
    predicted_stock_price = model.predict(X_val)
    
    #plot the loss, mae curve and stock_price curve
    plotter(History, '1', predicted_stock_price)
    
    