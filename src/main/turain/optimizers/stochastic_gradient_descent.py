from main.turain.optimizers.optimizer import Optimizer


class StochasticGradientDescent(Optimizer):
    def step(self):
        for parameter in self.parameters:
            if parameter.gradient is None:
                continue
            parameter.data = parameter.data - self.learning_rate * parameter.gradient
