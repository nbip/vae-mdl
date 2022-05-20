from typing import Any, NamedTuple

import tensorflow as tf


class Distribution(NamedTuple):
    distribution: Any
    sample: tf.Tensor
    axes: tuple = [-1, -2, -3]


class HierarchicalEncoder:
    def call(
        self, x: tf.Tensor, n_sample: int = 1, **kwargs
    ) -> dict[str, Distribution]:
        raise NotImplemented()


class HierarchicalDecoder:
    def call(self, Zs: dict[Distribution]) -> dict[str, Distribution]:
        raise NotImplemented()


class StochasticBlock:
    def call(self, x: tf.Tensor) -> Distribution:
        raise NotImplemented()


# TODO:
# Take model49 (2 stochastic layers) and make it more like model47
# in its build. Make it more modular so more stochastic layer can
# easily be added, mlp or conv, shouldn't matter
