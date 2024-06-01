import tensorflow as tf

import sys
sys.path.extend(['./','../','../Models/','../Data_Loader/','../Model_Helpers/'])

def dice_soft(y_true, y_pred, loss_type="sorensen", axis=(1, 2, 3), smooth=1e-5, from_logits=False):
    """
    Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.
    
    https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient
    """
    if not from_logits:
        _epsilon = tf.convert_to_tensor(1e-7, dtype=y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
        y_pred = tf.math.log(y_pred / (1 - y_pred))

    inse = tf.reduce_sum(y_pred * y_true, axis=axis)
    
    if loss_type == "jaccard":
        l = tf.reduce_sum(y_pred * y_pred, axis=axis)
        r = tf.reduce_sum(y_true * y_true, axis=axis)
    elif loss_type == "sorensen":
        l = tf.reduce_sum(y_pred, axis=axis)
        r = tf.reduce_sum(y_true, axis=axis)
    else:
        raise ValueError("Unknown loss_type")
        
    dice = (2.0 * inse + smooth) / (l + r + smooth)
    
    return tf.reduce_mean(dice)

def dice_hard(y_true, y_pred, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    """
    Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    The coefficient between 0 to 1, 1 if totally match.
    """
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true > threshold, tf.float32)
    
    inse = tf.reduce_sum(y_pred * y_true, axis=axis)
    l = tf.reduce_sum(y_pred, axis=axis)
    r = tf.reduce_sum(y_true, axis=axis)
    
    hard_dice = (2.0 * inse + smooth) / (l + r + smooth)
    
    return tf.reduce_mean(hard_dice)

def dice_loss(y_true, y_pred, from_logits=False):
    return 1 - dice_soft(y_true, y_pred, from_logits=from_logits)

def weighted_binary_crossentropy_loss(pos_weight):
    def loss(target, output, from_logits=False):
        if not from_logits:
            _epsilon = tf.convert_to_tensor(1e-7, dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.math.log(output / (1 - output))
        return tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=output, pos_weight=pos_weight)
    return loss

def margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0):
    def loss(labels, raw_logits):
        logits = raw_logits - 0.5
        positive_cost = pos_weight * labels * tf.cast(tf.less(logits, margin), tf.float32) * tf.pow(logits - margin, 2)
        negative_cost = (1 - labels) * tf.cast(tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
        return 0.5 * positive_cost + downweight * 0.5 * negative_cost
    return loss