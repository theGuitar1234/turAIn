from main.turain.optimizers.optimizer import Optimizer
from utilities import core_method
from utilities import TrainDefaults
from lib import override_from_parent

class Momentum(Optimizer):
    def __init__(self, parameters, backend, learning_rate=None, momentum_coefficient=None):
        super().__init__(parameters, learning_rate)
        if momentum_coefficient is None:
            momentum_coefficient = TrainDefaults.momentum_coefficient
        self.momentum_coefficient = momentum_coefficient
        self.backend = backend
        
        self.velocity = []
        for parameter in self.parameters:
            self.velocity.append(self.backend.zeros(parameter.data))
    
    @override_from_parent
    def step(self):
        for index, parameter in enumerate(self.parameters):
            if parameter.gradient is None:
                continue
            self.velocity[index] = (
                self.momentum_coefficient * self.velocity[index]
                + (1.0 - self.momentum_coefficient) * parameter.gradient 
            )
            
            self.update_parameter(parameter, self.velocity[index], self.learning_rate)
    
    @override_from_parent
    def update_parameter(self, parameter, gradient, learning_rate):
        return super().update_parameter(parameter, gradient, learning_rate)
            
    