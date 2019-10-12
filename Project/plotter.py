# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:37:45 2019

"""

##plotter
import matplotlib.pyplot as plt
import numpy as np

def plotter(History):
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
    fig_acc = plt.figure(figsize=(4,4))
    plt.title("Accuracy Learning Curve")
    plt.plot(History.history["binary_accuracy"], label="binary_accuracy")
    plt.plot(History.history["val_binary_accuracy"], label="val_binary_accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend();



    return fig_loss, fig_acc