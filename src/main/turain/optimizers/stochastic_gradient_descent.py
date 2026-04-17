from main.turain.optimizers.optimizer import Optimizer


class StochasticGradientDescent(Optimizer):
    def step(self):
        raise NotImplementedError
