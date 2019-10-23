from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.metrics import jaccard_similarity_score


def weighted_bce_loss(weight_map, weight_strength):
    def weighted_bce(y_true, y_pred):
        weight_f = weight_map * weight_strength +1.
        wy_true_f = weight_f * y_true
        wy_pred_f = weight_f * y_pred
        return K.mean(K.binary_crossentropy(wy_true_f, wy_pred_f))
    return weighted_bce

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def jaccard(y_true, y_pred):
    return jaccard_similarity_score(y_true, y_pred, normalize=True, sample_weight=None)

"""def jaccard_distance(y_true, y_pred, smooth=100):
     Calculates mean of Jaccard distance as a loss function 
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd =  (1 - jac) * smooth
    return tf.reduce_mean(jd)"""


def jaccard_loss(y_true, y_pred):
    smooth=100
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth