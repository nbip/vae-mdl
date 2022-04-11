import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utils import logmeanexp


# Adding a property to all tfd.Distrubtion distributions
# https://stackoverflow.com/a/54522123
# Should be run when importing the below loss functions.
# See test/test_tfd_distribution
@property
def axes(self):
    return self._axes


@axes.setter
def axes(self, axes):
    self._axes = axes


tfd.Distribution.axes = axes


def iwae_loss(x, z, pz, qzx, pxz):

    lpz = tf.reduce_sum(pz.log_prob(z), axis=pz.axes)

    lqzx = tf.reduce_sum(qzx.log_prob(z), axis=qzx.axes)

    lpxz = tf.reduce_sum(pxz.log_prob(x), axis=pxz.axes)

    log_w = lpxz + (lpz - lqzx)

    # logmeanexp over samples, average over batch
    iwae_elbo = tf.reduce_mean(logmeanexp(log_w, axis=0), axis=-1)
    bpd = -iwae_elbo / (tf.math.log(2.0) * 32.0 * 32.0 * 3.0)
    # bits_pr_dim:
    # https://github.com/rasmusbergpalm/vnca/blob/main/modules/vnca.py#L185
    # https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/model/losses.py#L146

    log_snis = tf.math.log_softmax(log_w)
    kl = -tf.reduce_mean(lpz - lqzx, axis=0)

    return -iwae_elbo, {
        "iwae_elbo": iwae_elbo,
        "bpd": bpd,
        "lpxz": lpxz,  # tf.reduce_logsumexp(lpxz + log_snis, axis=0),
        "lqzx": lqzx,
        "lpz": lpz,
        "kl": kl,
    }


def elbo_loss(x, z, pz, qzx, pxz):

    lpz = tf.reduce_sum(pz.log_prob(z), axis=pz.axes)

    lqzx = tf.reduce_sum(qzx.log_prob(z), axis=qzx.axes)

    lpxz = tf.reduce_sum(pxz.log_prob(x), axis=pxz.axes)

    log_w = lpxz + (lpz - lqzx)

    elbo = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)

    return -elbo, {"loss": -elbo, "lpxz": lpxz}
