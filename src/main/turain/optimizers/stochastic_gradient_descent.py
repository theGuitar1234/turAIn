from main.turain.optimizers.optimizer import Optimizer
from utilities import core_method
from lib import override_from_parent


class StochasticGradientDescent(Optimizer):
    def __init__(self, parameters, backend, learning_rate=None):
        super().__init__(parameters, learning_rate)
        self.backend = backend

    @override_from_parent
    def step(self):
        for parameter in self.parameters:
            if parameter.gradient is None:
                continue
            self.update_parameter(parameter, parameter.gradient, self.learning_rate)

    @override_from_parent
    def update_parameter(self, parameter, gradient, learning_rate):
        return super().update_parameter(parameter, gradient, learning_rate)



    
