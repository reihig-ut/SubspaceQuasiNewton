from base import Optimizer


class FirstOrderMethod(Optimizer):
    def __init__(self, f, der, x0, **hparams):
        super().__init__(f, x0, **hparams)
        self.der = der
        self.g = self.der(x0)


class GradientDescent(FirstOrderMethod):
    def step_size(self):
        if self.hparams["step_size"] == "line_search":
            return line_search(
                self.x,
                self.f,
                self.g,
                -self.g,
                self.hparams["ls_alpha"],
                self.hparams["ls_beta"],
            )
        elif isinstance(self.hparams["step_size"], float):
            return self.hparams["step_size"]
        else:
            raise ValueError("Invalid step_size")

    def update(self) -> dict:
        start = time.perf_counter()
        self.x -= self.step_size() * self.g
        elapsed
