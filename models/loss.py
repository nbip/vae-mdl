import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utils import logmeanexp


def iwae_loss(z, qzx, x, pxz):
    pz = tfd.Normal(0, 1)

    lpz = tf.reduce_sum(pz.log_prob(z), axis=[-1, -2, -3])

    lqzx = tf.reduce_sum(qzx.log_prob(z), axis=[-1, -2, -3])

    # for mdl: lpxz is already summed over channels
    # lpxz = tf.reduce_sum(pxz.log_prob(x), axis=[-1, -2])
    lpxz = tf.reduce_sum(pxz.log_prob(x), axis=[-1, -2, -3])

    log_w = lpxz + (lpz - lqzx)

    iwae_elbo = tf.reduce_mean(logmeanexp(log_w, axis=0), axis=-1)
    bpd = -iwae_elbo / (tf.math.log(2.0) * 32 * 32 * 3)
    # https://github.com/rasmusbergpalm/vnca/blob/main/modules/vnca.py#L185
    # https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/model/losses.py#L146

    snis = tf.math.log_softmax(log_w)
    kl = snis * (lpz - lqzx)

    return bpd, {
        "iwae_elbo": -iwae_elbo,
        "bpd": bpd,
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
