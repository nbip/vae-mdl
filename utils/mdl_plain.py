import tensorflow as tf
from tensorflow_probability import distributions as tfd

from utils.discretized_logistic import DiscretizedLogistic


class PixelMixtureDiscretizedLogistic(DiscretizedLogistic):
    """
    Mixture of discretized logistic distributions, specific to pixels, but not similar to PixelCNN++.

    The difference is NOT conditioning on observed x.

    The specificity to pixels comes from summing over the sub-pixel logprobs
    before weighing with the mixture logits. That is, there are a number
    of mixtures for each pixel, not for each sub-pixel.
    """

    def __init__(self, parameters, low=-1.0, high=1.0, levels=256.0):
        """
        Assume parameter shape:   [batch, h, w, ch, n_mix]
        Assume mix_logits shape:  [batch, h, w, n_mix]

        Note that mix_logits does not have the channels dimension, as the
        mixture weights are for the full pixel, not the sub-pixels.
        """
        loc, logscale, mix_logits = get_mixture_params(parameters)
        super(PixelMixtureDiscretizedLogistic, self).__init__(
            loc, logscale, low, high, levels
        )

        # ---- assume the mixture parameters are added as the last dimension, e.g. [batch, features, n_mix]
        self.mix_logits = mix_logits
        self.n_mix = self.mix_logits.shape[-1]

    def log_prob(self, x):
        """
        Mixture of discretized logistic distrbution log probabilities.

        Assume x shape:            [batch, h, w, ch]
        Assume parameter shape:    [batch, h, w, ch, n_mix]
        Assume mix_logits shape:   [batch, h, w, n_mix]
        """

        # ---- specific to this project: x from [0., 1.] to [-1., 1.]
        x = x * 2.0 - 1.0

        # ---- extend the last dimension of x to match the parameter shapes
        # ---- [batch, h, w, ch, n_mix]
        discretized_logistic_log_probs = super(
            PixelMixtureDiscretizedLogistic, self
        ).log_prob(x[..., None])

        # ---- convert mixture logits to mixture log weights
        # ---- [batch, h, w, n_mix]
        mix_log_weights = tf.nn.log_softmax(self.mix_logits, axis=-1)

        # ---- pixel-cnn style: sum over sub-pixel log_probs before mixture-weighing
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L83
        weighted_log_probs = (
            tf.reduce_sum(discretized_logistic_log_probs, axis=-2) + mix_log_weights
        )

        # ---- sum over weighted log-probs
        # ---- [batch, h, w, ch]
        return tf.reduce_logsumexp(weighted_log_probs, axis=-1)

    def sample(self, n_samples=[]):
        """OBS! this is not similar to the OpenAI PixelCNN sampling!

        See the original sampling method here: https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L89

        Instead, since there is no autoregressive dependence on x in p(r,g,b),
        there is no need to sample r,g,b sequentially.
        """

        # ---- sample the mixture component
        cat_dist = tfd.Categorical(logits=self.mix_logits)  # [batch, h, w]
        cat_samples = cat_dist.sample(n_samples)            # [n_samples, batch, h, w]
        cat_samples_onehot = tf.one_hot(cat_samples, axis=-1, depth=self.n_mix)  # [n_samples, batch, h, w, n_mix]
        # TODO: maybe reparameterizable?

        # ---- sample the logistic distributions
        # [n_samples, batch, h, w, ch=3]
        logistic_samples = super(PixelMixtureDiscretizedLogistic, self).sample(
            n_samples
        )

        # ---- pin out the samples chosen by the categorical distribution
        # we do that by multiplying the samples with a onehot encoding of the
        # mixture samples then summing along the last axis
        selected_samples = tf.reduce_sum(logistic_samples * cat_samples_onehot[..., None, :], axis=-1)

        # ---- specific to this project: samples from [-1., 1.] to [0., 1.]
        selected_samples = (selected_samples + 1.0) / 2.0

        return selected_samples

    def mean(self, **kwargs):
        # ---- sample the mixture component
        # TODO: average over softmax instead
        cat_dist = tfd.Categorical(logits=self.mix_logits)  # [batch, h, w]
        cat_samples = cat_dist.sample()  # [batch, h, w]
        cat_samples_onehot = tf.one_hot(cat_samples, axis=-1, depth=self.n_mix)  # [batch, h, w, n_mix]

        # ---- pin out the locs chosen by the categorical distribution
        # we do that by multiplying the locs with a onehot encoding of the
        # mixture samples then summing along the last axis
        selected_locs = tf.reduce_sum(self.loc * cat_samples_onehot[..., None, :], axis=-1)
        selected_locs = tf.clip_by_value(selected_locs, -1.0, 1.0)

        # ---- specific to this project: go from [-1., 1.] to [0., 1.]
        selected_locs = (selected_locs + 1.0) / 2.0
        return selected_locs


def get_mixture_params(parameters):
    """
    Prepare parameters for a mixture of discretized logistic distributions.

    Assumes parameters shape: [batch, h, w, n_mix * 10]
    Assumes x shape: [batch, h, w, n_channels = 3]
    Assumes x in [-1., 1.]

    :returns loc, logscale, mix_logits  # [batch, h, w, 3, n_mix]

    Each pixel location is modeled as
      (r,g,b) = (r)(g|r)(b|r,g)

    For each pixel there are n_mix * 10 parameters in total:
    - n_mix logits. These cover the whole pixel
    - n_mix * 3 loc. These are specific to each sub-pixel (r,g,b)
    - n_mix * 3 logscale. These are specific to each sub-pixel (r,g,b)
    - n_mix * 3 coefficients: 1 for p(g|r) and 2 for p(b|r,g). These are specific to each sub-pixel (r,g,b)
    """

    shape = parameters.shape
    n_mix = shape[-1] // 10

    # ---- get the mixture logits, for a full pixel there are n_mix logits (not 3 x n_mix)
    mix_logits = parameters[..., :n_mix]  # [batch, h, w, n_mix]

    # ---- reshape the rest of the parameters: [batch, h, w, 3 * 3 * n_mix] -> [batch, h, w, 3, 3 * n_mix]
    parameters = tf.reshape(parameters[..., n_mix:], shape[:-1] + [3, 3 * n_mix])

    # ---- split the rest of the parameters -> [batch, h, w, 3, n_mix]
    _loc, logscale, coeffs = tf.split(parameters, num_or_size_splits=3, axis=-1)
    logscale = tf.maximum(logscale, -7)
    coeffs = tf.nn.tanh(coeffs)

    # ---- adjust the locs, so instead of (r,g,b)
    # (r,g,b) = (r)(g|r)(b|r,g)
    loc_r = _loc[..., 0, :]
    loc_g = _loc[..., 1, :] + coeffs[..., 0, :] * loc_r
    loc_b = _loc[..., 2, :] + coeffs[..., 1, :] * loc_r + coeffs[..., 2, :] * loc_g

    loc = tf.concat(
        [loc_r[..., None, :], loc_g[..., None, :], loc_b[..., None, :]], axis=-2
    )

    return loc, logscale, mix_logits


if __name__ == '__main__':

    import os

    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---- generate torch / tf data
    b, c, h, w = 5, 3, 4, 4
    n_mixtures = 5
    x = np.random.rand(b, h, w, c).astype(np.float32)

    # bin the data, to resemble images
    bin = True
    if bin:
        x = np.floor(x * 256.0) / 255.0

    x = tf.convert_to_tensor(x)

    logits = np.random.randn(b, h, w, n_mixtures * 10).astype(np.float32)
    logits = tf.convert_to_tensor(logits)

    p = PixelMixtureDiscretizedLogistic(logits)
    lp = p.log_prob(2.0 * x - 1.0)
    print(lp.shape)
    print(p.sample(1000).shape)

    # ---- a leading sample dimension, as in IWAEs:
    s, b, c, h, w = 10, 6, 3, 4, 4
    n_mixtures = 5
    logits = np.random.randn(s, b, h, w, n_mixtures * 10).astype(np.float32)
    logits = tf.convert_to_tensor(logits)
    x = np.random.rand(b, h, w, c).astype(np.float32)

    # bin the data, to resemble images
    bin = True
    if bin:
        x = np.floor(x * 256.0) / 255.0

    x = tf.convert_to_tensor(x)

    p = PixelMixtureDiscretizedLogistic(logits)

    lp = p.log_prob(2.0 * x - 1.0)
    print(lp.shape)
    print(p.sample(1000).shape)
