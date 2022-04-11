"""
Show how to use tfd.Independent to handle .log_prob over several dimensions

https://www.tensorflow.org/probability/examples/TensorFlow_Distributions_Tutorial#using_independent_to_aggregate_batches_to_events
https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


class Normal(tfd.Normal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._axes = [-1]

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, axes):
        self._axes = axes


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---- data
    b, h, w, c = 5, 32, 32, 3
    x = tf.random.uniform([b, h, w, c], dtype=tf.float32)
    loc = tf.random.normal(x.shape)
    scale = tf.exp(tf.random.normal(x.shape))

    # ---- distribution
    p = tfd.Normal(loc, scale)

    # ---- straight forward way: just sum over the h, w, c dimensions
    lp = p.log_prob(x)
    print(lp.shape)
    print(tf.reduce_sum(lp, axis=[-1, -2, -3]))
    print("event shape, ", p.event_shape)

    # Issue: in some loss-function I have to know which axes to sum over
    # and might different in different models.
    # Solution: tfd.Independent specifies implicitly which axes should be summed
    p2 = tfd.Independent(p, reinterpreted_batch_ndims=3)

    # I don't get the naming here, but it seems that reinterpreted_batch_ndims
    # takes dimensions from the right and makes them events
    print("event shape, ", p2.event_shape)

    # Now we don't have to explicitly sum over the h, w, c dimensions
    print("tf.reduce_sum log_prob: \t", tf.reduce_sum(lp, axis=[-1, -2, -3]))
    print("tfd.Independent log_prob: \t", p2.log_prob(x))

    # Can we handle multiple leading dimensions?
    samples, b, h, w, c = 10, 5, 32, 32, 3
    x = tf.random.uniform([samples, b, h, w, c], dtype=tf.float32)
    loc = tf.random.normal(x.shape)
    scale = tf.exp(tf.random.normal(x.shape))

    p = tfd.Independent(tfd.Normal(loc, scale), reinterpreted_batch_ndims=3)
    print(p)

    # What about broadcasting?
    x = tf.random.uniform([b, h, w, c], dtype=tf.float32)
    loc = tf.random.normal([samples, b, h, w, c], dtype=tf.float32)
    scale = tf.exp(tf.random.normal([samples, b, h, w, c], dtype=tf.float32))

    p = tfd.Independent(tfd.Normal(loc, scale), reinterpreted_batch_ndims=3)
    print("batch and evetn shapes: ", p)
    print("x shape: ", x.shape)
    print("loc shape: ", p.mean().shape)
    print("p.log_prob(x) shape: ", p.log_prob(x).shape)

    # But subclassing tfd.Distribution is not straightforward,
    # how do I configure batch and event shapes?
    # Maybe instead write a wrapper around tfd distributions
    # which let's you set the axes to sum over?
    # Or should I subclass all relevant distributions and make
    # an attribute called "axes" which tells you how to sum?

    p = Normal(loc, scale)
    p.axes = [-1, -2, -3]
    tf.reduce_sum(p.log_prob(x), axis=p.axes).shape

    # A better way: https://stackoverflow.com/a/54522123
    # Works on all the tfd subclasses
    def set_axes(self, axes):
        self.axes = axes

    tfd.Distribution.set_axes = set_axes

    # tfd.Normal is subclassing tfd.Distribution
    p = tfd.Normal(loc, scale)
    p.set_axes([-1, -2, -3])
    print(p.axes)

    # tfd.Logistic is subclassing tfd.Distribution
    p = tfd.Logistic(loc, scale)
    p.set_axes([-1, -2, -3])
    print(p.axes)

    # But should I have that as a preamble in all my scripts?

    # Seems you don't even need getters and setters
    tfd.Distribution.testing = 42
    p = tfd.Logistic(loc, scale)
    print(p.testing)

    # but should you use getters and setters?
    # https://www.geeksforgeeks.org/getter-and-setter-in-python/
    # I think you should use setters to test logic

    # So maybe raise a value error if the axes haven't been set?

    @property
    def axes(self):
        print("Axes called")
        return self._axes

    @axes.setter
    def axes(self, axes):
        print("Setter called")
        self._axes = axes

    tfd.Distribution.axes = axes

    p = tfd.Logistic(loc, scale)
    p.axes = [-1, -2, -3]
    p.axes
