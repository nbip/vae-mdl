"""
Not a test as such, more like a minimal example
"""
from typing import NamedTuple, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from utils import DiscretizedLogistic, DistributionTuple, logmeanexp, setup_data


def loss_fn(
    x: tf.Tensor,
    Qs: dict[int, DistributionTuple],
    Ps: dict[int, DistributionTuple],
    pxz: tfp.distributions.Distribution,
    prior: DistributionTuple,
) -> Tuple[tf.Tensor, dict]:
    """
    :param x: image data [b, h, w, c]
    :return: -elbo, metrics
    """

    top_layer = max(Qs.keys())

    # ---- prior p(z)
    p, _, paxes = list(prior)
    q, z, qaxes = list(Qs[top_layer])
    log_p = tf.reduce_sum(p.log_prob(z), axis=paxes)
    log_q = tf.reduce_sum(q.log_prob(z), axis=qaxes)
    kl = [log_p - log_q]

    # ---- stochastic layers 1 : L-1
    for i in range(1, top_layer):
        q, z, qaxes = list(Qs[i])
        p, _, paxes = list(Ps[i])

        log_q = tf.reduce_sum(q.log_prob(z), axis=qaxes)
        log_p = tf.reduce_sum(p.log_prob(z), axis=paxes)
        kl.append(log_p - log_q)

    # ---- observation model p(x | z_1)
    lpxz = tf.reduce_sum(pxz.log_prob(x), axis=[-1, -2, -3])

    # ---- log weights
    log_w = lpxz + tf.add_n(kl)

    # ---- logmeanexp over samples, average over batch
    iwae_elbo = tf.reduce_mean(logmeanexp(log_w, axis=0), axis=-1)

    # ---- bits_pr_dim:
    # https://github.com/rasmusbergpalm/vnca/blob/main/modules/vnca.py#L185
    # https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/model/losses.py#L146
    n_dims = tf.cast(tf.math.reduce_prod(x.shape[1:]), tf.float32)
    bpd = -iwae_elbo / (tf.math.log(2.0) * n_dims)

    return -iwae_elbo, {"iwae_elbo": iwae_elbo, "bpd": bpd, "lpxz": lpxz, "kl": kl}


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    s, b, h, w, c = 5, 16, 32, 32, 3

    x = tf.random.uniform([b, h, w, c])

    Qs = {}
    Ps = {}
    prior = DistributionTuple(tfd.Normal(0.0, 1.0), None, [-1, -2, -3])
    pxz = DiscretizedLogistic(
        tf.random.uniform([s, b, h, w, c]), tf.exp(tf.random.normal([s, b, h, w, c]))
    )

    n_layers = 3
    for i in range(1, n_layers + 1):
        h, w = h // 2, w // 2

        q = tfd.Normal(
            tf.random.normal([s, b, h, w, c]), tf.exp(tf.random.normal([s, b, h, w, c]))
        )
        z = q.sample()

        Qs[i] = DistributionTuple(q, z)

        if i == n_layers:
            continue
        Ps[i] = DistributionTuple(
            tfd.Normal(
                tf.random.normal([s, b, h, w, c]),
                tf.exp(tf.random.normal([s, b, h, w, c])),
            )
        )

    elbo = loss_fn(x, Qs, Ps, pxz, prior)
    print(elbo[0])
