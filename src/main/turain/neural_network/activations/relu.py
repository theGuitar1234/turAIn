from main.turain.neural_network.activations.activation import Activation
from lib import override_from_parent
from utilities import core_method


class ReLu(Activation):
    def __init__(self, logit):
        super().__init__()
        self.logit = logit

    @override_from_parent
    def activate(self, logit):
        xp = self.backend
        logit = xp.asarray(logit, dtype=float)
        return xp.maximum(0.0, logit)

    @override_from_parent
    def derivative(self, z):
        return z > 0

    @override_from_parent
    def forward_propagation(self, x):
        self.input_cache = x
        return self.activate(x)

    @override_from_parent
    def backward_propagation(self, gradient_output):
        x = self.input_cache
        gradient_input = gradient_output * self.derivative(x)
        return gradient_input

    @override_from_parent
    def parameters(self):
        return super().parameters()


if __name__ == "__main__":
    pass
