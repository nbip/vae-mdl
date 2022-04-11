import os

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from models.loss import iwae_loss

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---- data
    b, h, w, c = 5, 32, 32, 3
    x = tf.random.uniform([b, h, w, c], dtype=tf.float32)
    loc = tf.random.normal(x.shape)
    scale = tf.exp(tf.random.normal(x.shape))

    # ---- importing iwae_loss should add the "axes" property to tfd.Distribution

    p = tfd.Normal(loc, scale)
    p.axes = [-1, -2, -3]
    print(p.axes)

    # ---- intended use
    print("x shape: \t", x.shape)
    print("p.log_prob(x) shape: \t", tf.reduce_sum(p.log_prob(x), axis=p.axes).shape)
