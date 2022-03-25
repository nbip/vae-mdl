import tensorflow as tf


def logmeanexp(log_w, axis):
    max = tf.reduce_max(log_w, axis=axis)
    return tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=axis)) + max
