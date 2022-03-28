import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utils import logmeanexp


def iwae_loss(z, qzx, x, pxz):
    pz = tfd.Normal(0, 1)

    lpz = tf.reduce_sum(pz.log_prob(z), axis=[-1, -2, -3])

    lqzx = tf.reduce_sum(qzx.log_prob(z), axis=[-1, -2, -3])

    lpxz = tf.reduce_sum(pxz.log_prob(x), axis=[-1, -2])
    # lpxz = tf.reduce_sum(pxz.log_prob(x), axis=[-1, -2, -3])

    log_w = lpxz + (lpz - lqzx)

    iwae_elbo = tf.reduce_mean(logmeanexp(log_w, axis=0), axis=-1)

    snis = tf.math.log_softmax(log_w)
    kl = snis * (lpz - lqzx)

    return -iwae_elbo, {
        "loss": -iwae_elbo,
        "lpxz": lpxz + snis,
        "lqzx": lqzx,
        "lpz": lpz,
        "kl": kl,
    }


def elbo_loss(z, qzx, x, pxz):
    pz = tfd.Normal(0, 1)

    lpz = tf.reduce_sum(pz.log_prob(z), axis=[-1, -2, -3])

    lqzx = tf.reduce_sum(qzx.log_prob(z), axis=[-1, -2, -3])

    lpxz = tf.reduce_sum(pxz.log_prob(x), axis=[-1, -2])

    log_w = lpxz + (lpz - lqzx)

    elbo = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)

    return -elbo, {"loss": -elbo, "lpxz": lpxz}
