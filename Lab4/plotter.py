# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:37:45 2019

"""

##plotter
import matplotlib.pyplot as plt
import numpy as np

def plotter(History, recall = False, precision = False):
    #Training vs Validation Learning loss 
    fig_loss = plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend(); 

    
    #Train and test accuracy plot
    fig_dice = plt.figure(figsize=(4,4))
    plt.title("Dice Score Curve")
    plt.plot(History.history["dice_coef"], label="dice_coef")
    plt.plot(History.history["val_dice_coef"], label="val_dice_coef")
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coef')
    plt.legend(); 


    return fig_loss, fig_dice