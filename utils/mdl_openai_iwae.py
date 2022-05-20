"""
Wrapper around the OpenAI pixelCNN implementation,
with functionality for leading sample dimensions, as in the IWAE

https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
https://github.com/rasmusbergpalm/vnca/blob/main/modules/dml.py
https://github.com/NVlabs/NVAE/blob/master/distributions.py#L120
https://github.com/openai/vdvae/blob/main/vae_helpers.py
"""
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import reparameterization


class MixtureDiscretizedLogisticOpenaiIWAE(tfd.Distribution):
    """
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution
    """

    def __init__(self, logits):
        super().__init__(
            dtype=logits.dtype,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=False,
        )

        self.logits = logits  # [n_samples] + [batch, h, w, n_mix * 10]
        self.shape = logits.shape
        self.n_mix = self.shape[-1] // 10

    def _log_prob(self, x):
        # ---- specific to this project: x from [0., 1.] to [-1., 1.]
        x = x * 2.0 - 1.0

        # ---- merge possible leading dimensions and batch dimensions
        logits_reshaped = tf.reshape(self.logits, [-1, *self.shape[-3:]])
        # Note that this reshape will manifest as
        # [0 1 2
        #  0 1 2
        #  0 1 2
        #  0 1 2]  ->
        # [0 1 2 0 1 2 0 1 2 0 1 2]
        # where height is samples and width is batch.
        # So the in logits_reshaped, the batch index changes while the sample index is constant, until a new sample index

        # ---- repeat x to match the logits
        repeats = logits_reshaped.shape[0] // x.shape[0]
        # Note that a repeat like this:
        # tf.repeat(x, repeats=repeats, axis=0)
        # will manifest as
        # [0 1 2]  ->  [0 0 0 1 1 1 2 2 2]
        # which does not match the logits_reshape. Instead
        x_repeat = tf.repeat(x[None, :], repeats=repeats, axis=0)
        x_reshaped = tf.reshape(x_repeat, [-1, *x.shape[-3:]])

        # ---- get the logprob
        # [n_samples * batch, h, w]
        lp = discretized_mix_logistic_loss(x_reshaped, logits_reshaped, sum_all=False)

        # ---- unmerge the sample and batch dimensions
        # [n_samples, batch, h, w]
        lp_unmerged = tf.reshape(lp, self.shape[:-1])

        # ---- expand last (channel) dimension to be similar to other loss functions
        return tf.expand_dims(lp_unmerged, axis=-1)

    def _sample_n(self, n, seed=None, **kwargs):
        # https://github.com/tensorflow/probability/blob/v0.16.0/tensorflow_probability/python/distributions/logistic.py#L160

        # ---- merge possible leading dimensions and batch dimensions
        # [n_samples * batch, h, w, n_mix * 10]
        logits_reshaped = tf.reshape(self.logits, [-1, *self.shape[-3:]])

        # ---- expand logits to have [n] as the leading dimension
        # [n, batch * n_samples, h, w, n_mix * 10]
        n_logits = tf.repeat(tf.expand_dims(logits_reshaped, axis=0), repeats=n, axis=0)

        # ---- merge the sample dimension into the batch dimension
        # [n * batch * n_samples, h, w, n_mix * 10]
        n_logits_reshaped = tf.reshape(
            n_logits, [logits_reshaped.shape[0] * n, *self.logits.shape[-3:]]
        )

        # ---- get samples
        # [n * batch * n_samples, h, w, 3]
        samples = sample_from_discretized_mix_logistic(n_logits_reshaped, self.n_mix)

        # ---- unmerge the sample and batch + n_samples dimensions
        # [n, batch * n_samples, h, w, 3]
        samples_unmerged = tf.reshape(samples, [n, *logits_reshaped.shape[:-1], 3])

        # ---- unmerge batch and n_samples dimensions
        samples_unmerged2 = tf.reshape(
            samples_unmerged, [n, *self.logits.shape[:-1], 3]
        )

        return samples_unmerged2 * 0.5 + 0.5

    def _mean(self, n=100, **kwargs):
        return tf.reduce_mean(self.sample(n), axis=0)


# ---- Modified from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py


def int_shape(x):
    return list(map(int, x.get_shape()))


def log_sum_exp(x):
    """numerically stable log_sum_exp implementation that prevents overflow"""
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.math.log(tf.reduce_sum(tf.exp(x - m2), axis))


def log_prob_from_logits(x):
    """numerically stable log_softmax implementation that prevents overflow"""
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keepdims=True)
    return x - m - tf.math.log(tf.reduce_sum(tf.exp(x - m), axis, keepdims=True))


def discretized_mix_logistic_loss(x, l, sum_all=True):
    """log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval"""
    xs = int_shape(x)  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l)  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(
        ls[-1] / 10
    )  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = tf.maximum(l[:, :, :, :, nr_mix : 2 * nr_mix], -7.0)
    coeffs = tf.nn.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(
        xs + [nr_mix]
    )  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(
        means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :],
        [xs[0], xs[1], xs[2], 1, nr_mix],
    )
    m3 = tf.reshape(
        means[:, :, :, 2, :]
        + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :]
        + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :],
        [xs[0], xs[1], xs[2], 1, nr_mix],
    )
    means = tf.concat(
        [tf.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], 3
    )
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(
        plus_in
    )  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(
        min_in
    )  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = (
        mid_in - log_scales - 2.0 * tf.nn.softplus(mid_in)
    )  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(
        x < -0.999,
        log_cdf_plus,
        tf.where(
            x > 0.999,
            log_one_minus_cdf_min,
            tf.where(
                cdf_delta > 1e-5,
                tf.math.log(tf.maximum(cdf_delta, 1e-12)),
                log_pdf_mid - np.log(127.5),
            ),
        ),
    )

    log_probs = tf.reduce_sum(log_probs, 3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        # return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])
        return log_sum_exp(log_probs)


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    # sel = tf.one_hot(tf.argmax(logit_probs - tf.math.log(-tf.math.log(tf.random.uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    # sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # or
    # sel = tf.one_hot(tfd.Categorical(logits=logit_probs).sample(), depth=nr_mix)
    # sel = tf.expand_dims(sel, axis=-2)
    # or
    sel = tfd.OneHotCategorical(logits=logit_probs, dtype=tf.float32).sample()
    sel = tf.expand_dims(sel, axis=-2)

    # select logistic parameters
    means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, 4)
    log_scales = tf.maximum(
        tf.reduce_sum(l[:, :, :, :, nr_mix : 2 * nr_mix] * sel, 4), -7.0
    )
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix]) * sel, 4)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    # u = tf.random.uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    # x = means + tf.exp(log_scales)*(tf.math.log(u) - tf.math.log(1. - u))
    # or
    x = tfd.Logistic(means, tf.exp(log_scales)).sample()

    x0 = tf.minimum(tf.maximum(x[:, :, :, 0], -1.0), 1.0)
    x1 = tf.minimum(tf.maximum(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.0), 1.0)
    x2 = tf.minimum(tf.maximum(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.0), 1.0)  # fmt: skip
    return tf.concat([tf.reshape(x0, xs[:-1] + [1]), tf.reshape(x1, xs[:-1] + [1]), tf.reshape(x2, xs[:-1] + [1])], 3)  # fmt:skip


if __name__ == "__main__":

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---- generate torch / tf data
    s, b, h, w, c = 17, 5, 4, 4, 3
    n_mixtures = 5
    x = np.random.rand(b, h, w, c).astype(np.float32)

    # bin the data, to resemble images
    bin = True
    if bin:
        x = np.floor(x * 256.0) / 255.0

    x = tf.convert_to_tensor(x)

    logits = np.random.randn(s, b, h, w, n_mixtures * 10).astype(np.float32)
    logits = tf.convert_to_tensor(logits)

    p = MixtureDiscretizedLogisticOpenaiIWAE(logits)

    print(p.log_prob(2.0 * x - 1.0).shape)
    print(p.sample().shape)
    print(p.sample(10).shape)
