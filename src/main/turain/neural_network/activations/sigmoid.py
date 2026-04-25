from lib import override_from_parent
from main.turain.neural_network.activations.activation import Activation
from utilities import core_method


class Sigmoid(Activation):
    def __init__(self, z):
        super().__init__()
        self.z = z

    @override_from_parent
    def activate(self, z):
        xp = self.backend.xp
        z = xp.asarray(z, dtype=float)
        output = xp.empty_like(z)

        positives = z >= 0
        negatives = ~positives

        output[positives] = 1.0 / (1.0 + xp.exp(-z[positives]))
        exp_of_z = xp.exp(z[negatives])
        output[negatives] = exp_of_z / (1.0 + exp_of_z)

        return output

    @override_from_parent
    def derivative(self, a):
        return a * (1 - a)

    @override_from_parent
    def forward_propagation(self, x):
        return self.activate(x)

    @override_from_parent
    def backward_propagation(self, gradient_output):
        x = self.input_cache
        gradient_input = gradient_output * self.derivative(x)
        return gradient_input


if __name__ == "__main__":
    pass
