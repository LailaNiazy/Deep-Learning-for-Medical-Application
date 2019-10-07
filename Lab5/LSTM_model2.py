# -*- coding: utf-8 -*-


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Input, Masking
from tensorflow.keras.layers import Dense, Dropout,Activation, Bidirectional

def model_LSTM2(drop_rate, units, input_shape, masking):

    model = Sequential()

    if masking:
        model.add(Masking(mask_value=0., input_shape=input_shape))
        
        model.add(Bidirectional(LSTM(units, return_sequences = True)))
    else:
        model.add(Bidirectional(LSTM(units, return_sequences = True), input_shape = input_shape))
    model.add(Dropout(drop_rate))

    model.add(LSTM(units, return_sequences = True))
    model.add(Dropout(drop_rate))

    model.add(LSTM(units, return_sequences = True))
    model.add(Dropout(drop_rate))

    model.add(LSTM(units))
    model.add(Dropout(drop_rate))

    model.add(Dense(1))
    
    model.add(Activation('sigmoid'))
   
  
    
    return model