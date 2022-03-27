import tensorflow as tf
from tensorflow_probability import distributions as tfd


class DiscretizedLogistic:
    """
    Discretized version of the logistic distribution f(x; mu, s) = e^{-(x - mu) / s} / s(1 + e^{-(x-mu)/s})^2
    """

    def __init__(self, loc, logscale, low=-1.0, high=1.0, levels=256.0):
        self.loc = loc
        self.logscale = logscale
        self.low = low
        self.high = high
        self.levels = levels

        # ---- width of interval around each center-value
        self.interval_width = (high - low) / (levels - 1.0)

        # ---- half interval width for range edge cases
        self.dx = self.interval_width / 2.0

    def logistic_cdf(self, x):
        a = (x - self.loc) * tf.exp(-self.logscale)
        return tf.nn.sigmoid(a)

    def logistic_log_prob_approx(self, x):
        """
        log pdf value times interval width as an approximation to the area under the curve in that interval
        """
        a = (x - self.loc) / tf.exp(self.logscale)
        log_pdf_val = -a - self.logscale - 2 * tf.nn.softplus(-a)
        return log_pdf_val + tf.cast(tf.math.log(self.interval_width), tf.float32)

    def log_prob(self, x):

        centered_x = x - self.loc
        inv_std = tf.exp(-self.logscale)

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
        log_prob_approx = self.logistic_log_prob_approx(x)

        # ---- use tf.where to choose between the true prob or the approximation
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

    def sample(self, n_samples=[]):
        logistic_dist = tfd.Logistic(loc=self.loc, scale=tf.exp(self.logscale))
        samples = logistic_dist.sample(n_samples)
        samples = tf.clip_by_value(samples, self.low, self.high)

        return samples
