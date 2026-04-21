from main.turain.optimizers.optimizer import Optimizer
from utilities import TrainDefaults
from lib import override_from_parent


class RMSProp(Optimizer):
    def __init__(self, backend, parameters, learning_rate=None, rms_coefficient=None, epsilon=None):
        super().__init__(parameters, learning_rate)
        if rms_coefficient is None:
            rms_coefficient = TrainDefaults.rms_coefficient
        self.rms_coefficient = rms_coefficient
        if epsilon is None:
            epsilon = TrainDefaults.epsilon
        self.epsilon = TrainDefaults.epsilon
        self.backend = backend

        self.square_average = []
        for parameter in self.parameters:
            self.square_average.append(self.backend.zeros(parameter.data))

    @override_from_parent
    def step(self):
        for index, parameter in enumerate(self.parameters):
            if parameter.gradient is None:
                continue

            self.square_average[index] = self.rms_coefficient * self.square_average[index] + (
                1.0 - self.rms_coefficient
            ) * (parameter.gradient * parameter.gradient)

            self.update_parameter(parameter, parameter.gradient, self.learning_rate)

    @override_from_parent
    def update_parameter(self, parameter, gradient, learning_rate):
        return super().update_parameter(parameter, gradient, learning_rate)


if __name__ == "__main__":
    pass
