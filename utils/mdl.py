"""
Mixture of discretized logistic distributions

Rewritten for customizability, but with results equal
to the original OpenAI PixelCNN version, see the tests

https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
https://github.com/rasmusbergpalm/vnca/blob/main/modules/dml.py
https://github.com/NVlabs/NVAE/blob/master/distributions.py#L120
https://github.com/openai/vdvae/blob/main/vae_helpers.py
https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution
"""
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import reparameterization


class MixtureDiscretizedLogistic(tfd.Distribution):
    def __init__(self, parameters):
        """
        Mixture of discretized logistic distributions.

        Assumes parameters shape: [batch, h, w, n_mix * 10]

        For each pixel there are n_mix * 10 parameters in total:
        - n_mix logits. These cover the whole pixel
        - n_mix * 3 loc. These are specific to each sub-pixel (r,g,b)
        - n_mix * 3 logscale. These are specific to each sub-pixel (r,g,b)
        - n_mix * 3 coefficients: 1 for p(g|r) and 2 for p(b|r,g)

        This version is set up to be comparable to the original OpenAI pixelCNN version
        https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
        """
        super().__init__(
            dtype=parameters.dtype,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=False,
        )

        self._parameters = parameters
        self.shape = parameters.shape
        self.n_mix = self.shape[-1] // 10

        # ---- width of interval around each center-value
        self.interval_width = 2.0 / 255.0

        # ---- half interval width for range edge cases
        self.dx = self.interval_width / 2.0

        self.low, self.high = -1.0, 1.0

    def _log_prob(self, x):
        """
        Mixture of discretized logistic distribution log probabilities.

        Assume x shape:            [batch, h, w, ch]
        Assume parameter shape:    [batch, h, w, ch, n_mix * 10]
        """

        # ---- [batch, h, w, ch, n_mix]
        loc, logscale, mix_logits = self._get_autoregressive_params(x)

        # ---- extend the last dimension of x to match the parameter shapes
        # ---- [batch, h, w, ch, n_mix]
        discretized_logistic_log_probs = self._discretized_logistic_log_prob(
            x[..., None], loc, logscale
        )

        # ---- convert mixture logits to mixture log weights
        # ---- [batch, h, w, n_mix]
        mix_log_weights = tf.nn.log_softmax(mix_logits, axis=-1)

        # ---- pixel-cnn style: sum over sub-pixel log_probs before mixture-weighing
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L83
        # ---- [batch, h, w, n_mix]
        weighted_log_probs = (
            tf.reduce_sum(discretized_logistic_log_probs, axis=-2) + mix_log_weights
        )

        # ---- sum over mixture components
        # ---- [batch, h, w, ch]
        return tf.reduce_logsumexp(weighted_log_probs, axis=-1)

    def _split_params(self):
        """Split self._parameters into loc, logscale coeffs and mix_logits"""

        # ---- get the mixture logits. For a full pixel there are n_mix logits (not 3 x n_mix)
        mix_logits = self._parameters[..., : self.n_mix]  # [batch, h, w, n_mix]

        # ---- reshape the rest of the parameters: [batch, h, w, 3 * 3 * n_mix] -> [batch, h, w, 3, 3 * n_mix]
        parameters_reshaped = tf.reshape(
            self._parameters[..., self.n_mix :], self.shape[:-1] + [3, 3 * self.n_mix]
        )

        # ---- split the rest of the parameters -> [batch, h, w, 3, n_mix]
        _loc, logscale, coeffs = tf.split(
            parameters_reshaped, num_or_size_splits=3, axis=-1
        )
        logscale = tf.maximum(logscale, -7.0)
        coeffs = tf.nn.tanh(coeffs)

        return _loc, logscale, coeffs, mix_logits

    def _get_autoregressive_params(self, x):
        """
        Prepare parameters for a mixture of discretized logistic distributions, with autoregression in the pixel channels.

        Assumes parameters shape: [batch, h, w, n_mix * 10]
        Assumes x shape: [batch, h, w, ch = 3]
        Assumes x in [-1., 1.]

        :returns loc, logscale, mix_logits  # [batch, h, w, 3, n_mix]

        Each pixel is modeled as
        p(R,G,B) = p(R)p(G|R=r)p(B|R=r,G=g)
        conditioning on the actual pixel values for the red and green channels.

        For each pixel there are n_mix * 10 parameters in total:
        - n_mix logits. These cover the whole pixel
        - n_mix * 3 loc. These are specific to each sub-pixel (r,g,b)
        - n_mix * 3 logscale. These are specific to each sub-pixel (r,g,b)
        - n_mix * 3 coefficients: 1 for p(g|r) and 2 for p(b|r,g).
        """

        _loc, logscale, coeffs, mix_logits = self._split_params()

        # ---- adjust the locs, so instead of p(R,G,B) = p(R)p(G)p(B) we get
        # p(R,G,B) = p(R)p(G|R=r)p(B|R=r,G=g)
        loc_r = _loc[..., 0, :]
        loc_g = _loc[..., 1, :] + coeffs[..., 0, :] * x[..., 0, None]
        loc_b = (
            _loc[..., 2, :]
            + coeffs[..., 1, :] * x[..., 0, None]
            + coeffs[..., 2, :] * x[..., 1, None]
        )

        loc = tf.concat(
            [loc_r[..., None, :], loc_g[..., None, :], loc_b[..., None, :]], axis=-2
        )

        return loc, logscale, mix_logits

    def _logistic_cdf(self, x, loc, logscale):
        a = (x - loc) * tf.exp(-logscale)
        return tf.nn.sigmoid(a)

    def _logistic_log_prob_approx(self, x, loc, logscale):
        """
        log pdf value times interval width as an approximation to the area under the curve in that interval
        """
        a = (x - loc) * tf.exp(-logscale)
        log_pdf_val = -a - logscale - 2 * tf.nn.softplus(-a)
        return log_pdf_val + tf.cast(tf.math.log(self.interval_width), tf.float32)

    def _discretized_logistic_log_prob(self, x, loc, logscale):
        centered_x = x - loc
        inv_std = tf.exp(-logscale)

        # ---- Get the change in CDF in the interval [x - dx, x + dx]
        # Note that the order of subtraction matters here, with tolerance 1e-6
        # assert tf.reduce_sum((x - self.dx - self.loc)) == tf.reduce_sum((x - self.loc - self.dx)), 'Order of subtraction matters'
        interval_start = (centered_x - self.dx) * inv_std
        interval_stop = (centered_x + self.dx) * inv_std

        # ---- true probability based on the CDF
        prob = tf.nn.sigmoid(interval_stop) - tf.nn.sigmoid(interval_start)

        # ---- safeguard prob by taking the maximum of prob and 1e-12
        # this is only done to make sure tf.where does not fail
        prob = tf.math.maximum(prob, 1e-12)

        # ---- edge cases
        # Left edge, if x=-1.: All the CDF in ]-inf, x + dx]
        # Right edge, if x=1.: All the CDF in [x - dx, inf[
        left_edge = interval_stop - tf.nn.softplus(interval_stop)
        right_edge = -tf.nn.softplus(interval_start)

        # ---- approximated log prob, if the prob is too small
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L70
        log_prob_approx = self._logistic_log_prob_approx(x, loc, logscale)

        # ---- use tf.where to choose between the true prob or the approixmation
        safe_log_prob = tf.where(prob > 1e-5, tf.math.log(prob), log_prob_approx)

        # ---- use tf.where to select the edge case probabilities when relevant
        # if the input values are not binned, there is a difference between
        # using tf.less_equal(x, self.low) and x < -0.999 as in
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L81
        # otherwise there shouldn't be.
        safe_log_prob_with_left_edge = tf.where(
            tf.less_equal(x, self.low), left_edge, safe_log_prob
        )
        safe_log_prob_with_edges = tf.where(
            tf.greater_equal(x, self.high), right_edge, safe_log_prob_with_left_edge
        )

        return safe_log_prob_with_edges

    def _sample_n(self, n, seed=None, **kwargs):
        _loc, logscale, coeffs, mix_logits = self._split_params()

        # ---- sample from logistic distribution
        logistic_samples = tfd.Logistic(_loc, tf.exp(logscale)).sample(n)

        # ---- adjust the logistic samples, so instead of p(R,G,B) = p(R)p(G)p(B) we get
        # p(R,G,B) = p(R)p(G|R)p(B|R,G)
        # Notice we're not conditioning on any observed pixel values here
        sample_r = tf.clip_by_value(logistic_samples[..., 0, :], -1.0, 1.0)
        sample_g = tf.clip_by_value(
            logistic_samples[..., 1, :] + coeffs[..., 0, :] * sample_r, -1.0, 1.0
        )
        sample_b = tf.clip_by_value(
            logistic_samples[..., 2, :]
            + coeffs[..., 1, :] * sample_r
            + coeffs[..., 2, :] * sample_g,
            -1.0,
            1.0,
        )

        autoregressive_samples = tf.concat(
            [sample_r[..., None, :], sample_g[..., None, :], sample_b[..., None, :]],
            axis=-2,
        )

        # ---- sample mixture indicator from categorical distribution
        categorical_samples = tfd.OneHotCategorical(
            logits=mix_logits, dtype=tf.float32
        ).sample(n)
        # ---- expand pixel dimension
        categorical_samples = tf.expand_dims(categorical_samples, axis=-2)

        # ---- pin out the samples chosen by the categorical distribution
        # we do that by multiplying the logistic samples with onehot samples
        # from the mixture distribution
        selected_samples = tf.reduce_sum(
            autoregressive_samples * categorical_samples, axis=-1
        )

        return selected_samples

    def _mean(self, n=100, **kwargs):
        return tf.reduce_mean(self.sample(100), axis=0)


if __name__ == "__main__":

    import os

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

    p = MixtureDiscretizedLogistic(logits)
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

    p = MixtureDiscretizedLogistic(logits)

    lp = p.log_prob(2.0 * x - 1.0)
    print(lp.shape)
