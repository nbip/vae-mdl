import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd

from modules.residual import ResidualBlock


class Up2D(layers.Layer):
    """Make upscaling handle ndim = 5 instead of just ndim = 4"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.up = layers.UpSampling2D(*args, **kwargs)

    def call(self, x):
        shape = x.shape

        # merge sample and batch dimension
        x = tf.reshape(x, [-1, *shape[-3:]])

        out = self.up(x)

        # unmerge sample and batch dimension
        out = tf.reshape(out, [*shape[:-3], *out.shape[-3:]])

        return out


class DecoderBlock(layers.Layer):
    def __init__(self, hidden_width, out_width, n_blocks, upscale_rate, rezero=False):
        super().__init__()

        self.blocks = tf.keras.Sequential(
            [ResidualBlock(hidden_width, out_width, rezero) for _ in range(n_blocks)]
        )

        # self.interpolate = layers.UpSampling2D(size=(upscale_rate, upscale_rate), interpolation="bilinear")
        self.interpolate = Up2D(
            size=(upscale_rate, upscale_rate), interpolation="bilinear"
        )

    def call(self, x, **kwargs):
        x = self.interpolate(x)
        return self.blocks(x)


class StochasticDecoderBlock(layers.Layer):
    def __init__(self, hidden_width, out_width, n_blocks, upscale_rate, rezero=False):
        super().__init__()

        self.block = DecoderBlock(
            hidden_width, out_width, n_blocks, upscale_rate, rezero=rezero
        )
        self.conv = layers.Conv2D(
            out_width * 2,
            strides=1,
            kernel_size=3,
            padding="same",
            activation=tf.nn.gelu,
        )

    def call(self, x, **kwargs):
        x = self.block(x)
        mu, logstd = tf.split(self.conv(x), num_or_size_splits=2, axis=-1)
        return tfd.Normal(mu, tf.nn.softplus(logstd))


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    b, h, w, c = 128, 16, 16, 64
    hidden_width, out_width = 32, 64

    x = tf.random.uniform([b, h, w, c])
    block = DecoderBlock(hidden_width, out_width, n_blocks=3, upscale_rate=2)
    out = block(x)
    print(out.shape)

    b, h, w, c = 1, 16, 16, 4
    hidden_width, out_width = c // 2, c
    x = tf.Variable(np.random.rand(b, h, w, c).astype(np.float32))

    # ---- gradients summed over the output
    block = DecoderBlock(hidden_width, out_width, n_blocks=3, upscale_rate=2)
    with tf.GradientTape() as tape:
        out = block(x)
    grads = tape.gradient(out, x)
    fig, ax = plt.subplots()
    ax.imshow(grads[0, :, :, 0])
    plt.show()
    plt.close()

    # ---- receptive field for one outcome pixel
    block = DecoderBlock(hidden_width, out_width, n_blocks=3, upscale_rate=2)
    ix = [16, 16]
    with tf.GradientTape() as tape:
        out = block(x)[0, ix[0], ix[1], 0]
    grads = tape.gradient(out, x)
    fig, ax = plt.subplots()
    ax.imshow(np.ceil(np.abs(grads[0, :, :, 0])))
    plt.show()
    plt.close()

    # does it work with a sample dimension?
    s, b, h, w, c = 5, 128, 16, 16, 64
    hidden_width, out_width = c // 2, c

    x = tf.random.uniform([s, b, h, w, c])
    block = DecoderBlock(hidden_width, out_width, n_blocks=3, upscale_rate=2)
    out = block(x)
    print(out.shape)
