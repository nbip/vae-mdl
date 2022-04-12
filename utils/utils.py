import numpy as np
import tensorflow as tf


def logmeanexp(log_w, axis):
    max = tf.reduce_max(log_w, axis=axis)
    return tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=axis)) + max


def bernoullisample(x, seed=None):
    return tf.cast(
        tf.math.greater(x, tf.random.uniform(x.shape, seed=seed)), tf.float32
    )
