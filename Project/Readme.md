## Models:

##FR-Fa-GE

Configuration:
- base = 16
- batch_size = 1
- LR = 0.00001
- SDRate = 0.5
- batch_normalization = True
- spatial_dropout = True
- epochs = 150
- final_neurons= 1 #binary classification
- final_afun = "sigmoid" #activation function
- weight_strength = 1.
- Data augmentation
- Accuracy = jaccard coefficient
- Loss = weighted binary accuracy
- Optimizer = Adam

Result for the final epoch of the two last folds:

Epoch 150/150
77/77 [==============================] - 4s 50ms/step - loss: 0.0166 - jaccard_loss: 0.0049 - val_loss: 0.0215 - val_jaccard_loss: 0.0031

Epoch 150/150
77/77 [==============================] - 3s 43ms/step - loss: 0.0156 - jaccard_loss: 0.0050 - val_loss: 0.0241 - val_jaccard_loss: 0.0029