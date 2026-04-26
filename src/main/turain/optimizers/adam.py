from ..optimizers.optimizer import Optimizer
from ..lib import override_from_parent
from ..utilities import TrainDefaults


class Adam(Optimizer):
    def __init__(self, backend, parameters, config=None):
        if config is None:
            config = TrainDefaults()
        learning_rate = config.learning_rate
        momentum_coefficient = config.momentum_coefficient
        rms_coefficient = config.rms_coefficient
        epsilon = config.epsilon

        super().__init__(parameters, learning_rate)

        self.momentum_coefficient = momentum_coefficient
        self.rms_coefficient = rms_coefficient
        self.backend = backend
        self.epsilon = epsilon

        self.iteration = 0

        xp = self.backend.xp

        self.velocity = []
        self.square_average = []
        for parameter in self.parameters:
            zeros = xp.zeros(parameter.data)
            self.velocity.append(zeros)
            self.square_average.append(zeros)

    @override_from_parent
    def step(self):
        xp = self.backend.xp
        self.iteration += 1

        for index, parameter in enumerate(self.parameters):
            if parameter.gradient is None:
                continue

            self.velocity[index] = (
                self.momentum_coefficient * self.velocity[index]
                + (1.0 - self.momentum_coefficient) * parameter.gradient
            )

            self.second_moment[index] = self.rms_coefficient * self.square_average[index] + (
                1.0 - self.square_average
            ) * (parameter.gradient * parameter.gradient)

            momentum_bias_corrected = self.velocity[index] / (
                1.0 - self.momentum_coefficient**self.iteration
            )
            rms_bias_corrected = self.square_average[index] / (
                1.0 - self.rms_coefficient**self.iteration
            )

            gradient = momentum_bias_corrected / xp.square_root(rms_bias_corrected + self.epsilon)

            self.update_parameter(parameter, gradient, self.learning_rate)
