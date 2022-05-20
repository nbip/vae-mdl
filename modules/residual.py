import tensorflow as tf
from tensorflow.keras import layers


class ConvBuilder:
    @staticmethod
    def b1x1(out_dim):
        return layers.Conv2D(
            out_dim, kernel_size=1, strides=1, padding="same", activation=tf.nn.gelu
        )

    @staticmethod
    def b3x3(out_dim):
        return layers.Conv2D(
            out_dim, kernel_size=3, strides=1, padding="same", activation=tf.nn.gelu
        )


class ResidualBlock(layers.Layer):
    """https://github.com/vvvm23/vdvae/blob/main/vae.py#L62"""

    def __init__(self, hidden_width, out_width, rezero=False):
        super().__init__()
        self.conv = tf.keras.Sequential(
            [
                ConvBuilder.b1x1(hidden_width),
                ConvBuilder.b3x3(hidden_width),
                ConvBuilder.b3x3(hidden_width),
                ConvBuilder.b1x1(out_width),
            ]
        )

        self.gate = tf.Variable(0.0) if rezero else 1.0

    def call(self, x, **kwargs):
        return x + self.conv(x) * self.gate


if __name__ == "__main__":

    import os

    import matplotlib.pyplot as plt
    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    b, h, w, c = 128, 32, 32, 64
    hidden_width, out_width = 32, 64

    x = tf.random.uniform([b, h, w, c])

    block = ResidualBlock(hidden_width=hidden_width, out_width=out_width, rezero=True)

    out = block(x)
    print(out.shape)

    b, h, w, c = 1, 32, 32, 4
    hidden_width, out_width = c // 2, c
    x = tf.Variable(np.random.rand(b, h, w, c).astype(np.float32))

    # ---- gradients summed over the output
    block = ResidualBlock(hidden_width, out_width, rezero=False)
    with tf.GradientTape() as tape:
        out = block(x)
    grads = tape.gradient(out, x)
    fig, ax = plt.subplots()
    ax.imshow(grads[0, :, :, 0])
    plt.show()
    plt.close()

    # ---- receptive field for one outcome pixel
    block = ResidualBlock(hidden_width, out_width, rezero=False)
    ix = [14, 14]
    with tf.GradientTape() as tape:
        out = block(x)[0, ix[0], ix[1], 0]
    grads = tape.gradient(out, x)
    fig, ax = plt.subplots()
    ax.imshow(np.ceil(np.abs(grads[0, :, :, 0])))
    plt.show()
    plt.close()

    # ---- receptive field for one outcome pixel, for a stack
    stack = tf.keras.Sequential(
        [ResidualBlock(hidden_width, out_width, rezero=False) for _ in range(5)]
    )
    ix = [14, 14]
    with tf.GradientTape() as tape:
        out = stack(x)[0, ix[0], ix[1], 0]
    grads = tape.gradient(out, x)
    fig, ax = plt.subplots()
    ax.imshow(np.ceil(np.abs(grads[0, :, :, 0])))
    plt.show()
    plt.close()
