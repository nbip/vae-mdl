class Model:
    def train_batch(self) -> float:
        raise NotImplemented()

    def val_batch(self) -> float:
        raise NotImplemented()

    def save(self, fp: str) -> None:
        # TODO: https://www.tensorflow.org/guide/checkpoint
        raise NotImplemented()

    def load(self, fp: str) -> None:
        # TODO: how to load when model is not built yet?
        raise NotImplemented()
