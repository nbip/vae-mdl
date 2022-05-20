from collections import namedtuple
from typing import NamedTuple, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def logmeanexp(log_w, axis):
    max = tf.reduce_max(log_w, axis=axis)
    return tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=axis)) + max


def bernoullisample(x, seed=None):
    return tf.cast(
        tf.math.greater(x, tf.random.uniform(x.shape, seed=seed)), tf.float32
    )


class GlobalStep(object):
    """
    Functionality for updating a learning rate based on the global step

    https://stackoverflow.com/a/6192298
    https://codereview.stackexchange.com/q/253675
    """

    def __init__(self):
        self._value = 0
        self._observers = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        for callback in self._observers:
            # print('announcing change')
            callback(self._value)

    def bind_to(self, callback):
        # print('bound')
        self._observers.append(callback)


class Dist(namedtuple("Dist", "dist sample axes")):
    """
    Holds either variational or generative distribution data

    dist : tfd.Distribution
    samples : samples from dist
    axes : specifies axes to sum over in a loss
    """

    @property
    def z(self):
        return self.sample

    @property
    def x(self):
        return self.sample

    @property
    def p(self):
        return self.dist

    @property
    def q(self):
        return self.dist


def fill_canvas(img, n, h, w, c):
    canvas = np.empty([n * h, n * w, c])
    for i in range(n):
        for j in range(n):
            canvas[i * h : (i + 1) * h, j * w : (j + 1) * w, :] = img[i * n + j, :, :, :]  # fmt: skip

    return canvas


class DistributionTuple(NamedTuple):
    """collection of distribution, samples and reduction axes"""

    dist: tfp.distributions.Distribution
    sample: Optional[tf.Tensor] = None
    axes: tuple = (-1, -2, -3)

    @property
    def z(self):
        return self.sample

    @property
    def x(self):
        return self.sample
