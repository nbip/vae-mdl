import tensorflow as tf
from tensorflow.keras import layers


class GLU(layers.Layer):
    """GLU: https://arxiv.org/pdf/1612.08083.pdf"""

    def __init__(self, filters=64, activation=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)

        self.conv = tf.keras.Sequential(
            [
                layers.Conv2D(
                    filters,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                layers.Conv2D(
                    2 * filters,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation=None,
                ),
            ]
        )

    def call(self, inputs, **kwargs):
        A, B = tf.split(self.conv(inputs), 2, axis=-1)
        H = A * tf.nn.sigmoid(B)
        return tf.nn.relu(H)


if __name__ == "__main__":

    import os

    import matplotlib.pyplot as plt
    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    b, h, w, c = 1, 32, 32, 1
    x = np.random.rand(b, h, w, c).astype(np.float32)
    x = tf.Variable(x)

    # ---- gradients summed over the output
    glu = GLU()
    with tf.GradientTape() as tape:
        out = glu(x)
    grads = tape.gradient(out, x)
    fig, ax = plt.subplots()
    ax.imshow(grads[0])
    plt.show()
    plt.close()

    # ---- receptive field for one outcome pixel
    glu = GLU()
    ix = [14, 14]
    with tf.GradientTape() as tape:
        out = glu(x)[0, ix[0], ix[1]]
    grads = tape.gradient(out, x)
    fig, ax = plt.subplots()
    ax.imshow(np.abs(grads[0]))
    plt.show()
    plt.close()

    # ---- receptive field for one outcome pixel, for a stack
    stack = tf.keras.Sequential([GLU() for _ in range(5)])
    ix = [14, 14]
    with tf.GradientTape() as tape:
        out = stack(x)[0, ix[0], ix[1]]
    grads = tape.gradient(out, x)
    fig, ax = plt.subplots()
    ax.imshow(np.abs(grads[0]))
    plt.show()
    plt.close()
