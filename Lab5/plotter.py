# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:36:08 2019

@author: looly
"""

import matplotlib.pyplot as plt
import numpy as np

def plotter(History, task, predicted_stock_price = []):
    #Training vs Validation Learning loss 
    fig = plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend(); 
    fig.savefig('Plots/Loss_Curve_{}.png'.format(task))

    if task == '1':
        
        #Train and test accuracy plot
        fig2 = plt.figure(figsize=(4,4))
        plt.title("Mean Absolute Erro Curve")
        plt.plot(History.history["mean_absolute_error"], label="mean_absolute_error")
        plt.plot(History.history["val_mean_absolute_error"], label="val_mean_absolute_error")
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend(); 
        fig2.savefig('Plots/Mean_Absolute_Error_Curve_{}.png'.format(task))
     
        fig3 = plt.figure(figsize=(4,4))
        plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
        plt.title('TATA Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('TATA Stock Price')
        plt.legend()
        plt.show()
        fig3.savefig('Plots/predicted_stock_price_{}.png'.format(task))
        
    elif task == '3':
        #Train and test accuracy plot
        fig2 = plt.figure(figsize=(4,4))
        plt.title("Mean Absolute Erro Curve")
        plt.plot(History.history["dice_coef"], label="dice_coef")
        plt.plot(History.history["val_dice_coef"], label="val_dice_coef")
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend(); 
        fig2.savefig('Plots/Dice_Coef_Curve_{}.png'.format(task))
