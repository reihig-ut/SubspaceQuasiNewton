import mlflow
import numpy as np


class Optimizer:
    """Base class for optimization algorithms

    Args:
        f (Objective): objective function
        x0 (np.ndarray): initial point
    """

    def __init__(self, f, x0, **hparams):
        self.f = f
        self.x = x0
        self.terminate = False
        self.hparams = hparams

    def update(self) -> dict:
        """update x and return metrics

        Raises:
            NotImplementedError

        Returns:
            dict: {metric_name: metric_value}
        """
        raise NotImplementedError

    @staticmethod
    def log_metric(i: int, metrics: dict) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=i)

    @staticmethod
    def log_iterate(x: np.ndarray, path: str) -> None:
        np.save(path, x)
        mlflow.log_artifact(path)

    def log_hparam(self) -> None:
        mlflow.set_tag("optimizer", self.__class__.__name__)
        mlflow.set_tag("objective", self.f.__class__.__name__)
        self.log_iterate(self.x, "x_init.npy")
        for key, value in self.hparams.items():
            mlflow.log_param(key, value)

    def run(self, iteration):
        with mlflow.start_run():
            self.log_hparam()

            for i in range(iteration):
                metrics = self.update()
                self.log_metric(i + 1, metrics)
                if self.terminate:
                    break

            self.log_iterate(self.x, "x_last.npy")
