from tensorflow.keras import backend as K
import tensorflow as tf

def weighted_bce_loss(weight_map, weight_strength):
    def weighted_bce(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        weight_f = K.flatten(weight_map)
        weight_f = tf.cast(weight_f, tf.float32)
        weight_f = weight_f * weight_strength +1.
        wy_true_f = weight_f * y_true_f
        wy_pred_f = weight_f * y_pred_f
        return K.binary_crossentropy(wy_true_f, wy_pred_f)
    return weighted_bce