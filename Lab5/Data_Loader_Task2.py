# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np

tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)

##image loader
def data_loader_task2(dataPath, train_subjects_list,val_subjects_list, bundles_list, n_tracts_per_bundle):
    X_train, y_train = load_streamlines(dataPath, train_subjects_list, bundles_list,n_tracts_per_bundle) 
    X_val, y_val = load_streamlines(dataPath, val_subjects_list, bundles_list, n_tracts_per_bundle) 
    return X_train, y_train, X_val, y_val

import nibabel as nib 

def load_streamlines(dataPath, subject_ids, bundles, n_tracts_per_bundle): 
    X = [] 
    y = [] 
    
    for i in range(len(subject_ids)): 
        for c in range((len(bundles))): 
            filename = dataPath + subject_ids[i] + '/' + bundles[c] + '.trk' 
            tfile = nib.streamlines.load(filename) 
            streamlines = tfile.streamlines 
            
            n_tracts_total = len(streamlines) 
            ix_tracts = np.random.choice(range(n_tracts_total), n_tracts_per_bundle, replace=False) 
            streamlines_data = streamlines.data 
            streamlines_offsets = streamlines._offsets 
            
            for j in range(n_tracts_per_bundle): 
                ix_j = ix_tracts[j] 
                offset_start = streamlines_offsets[ix_j] 
                if ix_j < (n_tracts_total - 1): 
                    offset_end = streamlines_offsets[ix_j + 1] 
                    streamline_j = streamlines_data[offset_start:offset_end] 
                else: 
                    streamline_j = streamlines_data[offset_start:]
                X.append(np.asarray(streamline_j)) 
                y.append(c) 
   
    return X, y
