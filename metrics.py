import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="builtins")
def intersection_over_union(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        iou_score = (intersection + 1e-15) / (union + 1e-15)
        iou_score = iou_score.astype(np.float32)
        return iou_score
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15

@register_keras_serializable(package="builtins")
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

@register_keras_serializable(package="builtins")
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

@register_keras_serializable(package="builtins")
def binary_crossentropy_dice_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)

@register_keras_serializable(package="builtins")
def weighted_f_score(y_true, y_pred, beta=2):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positive = tf.reduce_sum(y_true * y_pred)
    false_positive = tf.reduce_sum((1 - y_true) * y_pred)
    false_negative = tf.reduce_sum(y_true * (1 - y_pred))
    precision = true_positive / (true_positive + false_positive + 1e-15)
    recall = true_positive / (true_positive + false_negative + 1e-15)
    f_score = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall + 1e-15)
    return f_score

@register_keras_serializable(package="builtins")
def s_score(y_true, y_pred, alpha=0.5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    true_positive = tf.reduce_sum(y_true * y_pred)
    false_positive = tf.reduce_sum((1 - y_true) * y_pred)
    false_negative = tf.reduce_sum(y_true * (1 - y_pred))
    s_object = true_positive / (true_positive + false_negative + 1e-15)
    s_region = true_positive / (true_positive + false_positive + 1e-15)
    return alpha * s_object + (1 - alpha) * s_region

@register_keras_serializable(package="builtins")
def e_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    true_positive = tf.reduce_sum(y_true * y_pred)
    false_positive = tf.reduce_sum((1 - y_true) * y_pred)
    false_negative = tf.reduce_sum(y_true * (1 - y_pred))
    precision = true_positive / (true_positive + false_positive + 1e-15)
    recall = true_positive / (true_positive + false_negative + 1e-15)
    return 2 * precision * recall / (precision + recall + 1e-15)

@register_keras_serializable(package="builtins")
def max_e_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    true_positive = tf.reduce_sum(y_true * y_pred)
    false_positive = tf.reduce_sum((1 - y_true) * y_pred)
    false_negative = tf.reduce_sum(y_true * (1 - y_pred))
    precision = true_positive / (true_positive + false_positive + 1e-15)
    recall = true_positive / (true_positive + false_negative + 1e-15)
    f_score = 2 * precision * recall / (precision + recall + 1e-15)
    return tf.reduce_max(f_score)

@register_keras_serializable(package="builtins")
def mean_absolute_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs(y_pred - y_true))