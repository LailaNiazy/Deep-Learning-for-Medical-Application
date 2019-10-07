

import matplotlib.pyplot as plt
import numpy as np

def plotter2(History, task):
   # fig = plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
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

    #Train and test accuracy plot
    fig2 = plt.figure(figsize=(4,4))
    plt.title("Binary Cross Entropy Curve")
    plt.plot(History.history["binary_accuracy"], label="binary_accuracy")
    plt.plot(History.history["val_binary_accuracy"], label="val_binary_accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(); 
    fig2.savefig('Plots/Binary_Accuracy_Curve_{}.png'.format(task))
    
    
    return fig, fig2