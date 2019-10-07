from tensorflow.keras.utils import Sequence 
import numpy as np

class MyBatchGenerator(Sequence): 
    def __init__(self, X, y, batch_size=1, shuffle = True): 
        #'Initialization' 
        self.X = X
        self.y = y 
        self.batch_size = batch_size 
        self.shuffle = shuffle 
        self.on_epoch_end() 
        
    def __len__(self): 
        #'Denotes the number of batches per epoch' 
        return int(np.floor(len(self.y)/self.batch_size)) 

    def __getitem__(self, index): 
        return self.__data_generation(index) 

    def on_epoch_end(self): 
       # 'Shuffles indexes after each epoch' 
        self.indexes = np.arange(len(self.y)) 
        if self.shuffle == True: 
            np.random.shuffle(self.indexes) 

    def __data_generation(self, index): 
        Xb = np.empty((self.batch_size, *self.X[index].shape)) 
        yb = np.empty((self.batch_size, 1))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size): 
            Xb[s] = self.X[index] 
            yb[s] = self.y[index] 
        return Xb, yb