# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:36:08 2019

@author: looly
"""

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras.layers import Dense, Dropout

def model_LSTM(drop_rate, units, time_sequence):

    model = Sequential()

    model.add(LSTM(units, return_sequences = True, input_shape = (time_sequence, 1)))
    model.add(Dropout(drop_rate))

    model.add(LSTM(units, return_sequences = True))
    model.add(Dropout(drop_rate))

    model.add(LSTM(units, return_sequences = True))
    model.add(Dropout(drop_rate))

    model.add(LSTM(units))
    model.add(Dropout(drop_rate))

    model.add(Dense(1))
    
    return model