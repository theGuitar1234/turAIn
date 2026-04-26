from ..utilities import TrainDefaults
from ..utilities import core_method

class Optimizer:
    def __init__(self, parameters, learning_rate=None):
        self.parameters = list(parameters)
        if learning_rate is None:
            learning_rate = TrainDefaults.learning_rate
        self.learning_rate = learning_rate

    @core_method
    def step(self):
        raise NotImplementedError
    
    def update_parameter(self, parameter, gradient, learning_rate):
        parameter.data = parameter.data - learning_rate * gradient

    def zero_gradient(self):
        for p in self.parameters:
            p.gradient = None