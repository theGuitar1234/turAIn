from main.turain.neural_network.activations.activation import Activation
from lib import override_from_parent


class HiperbolicTangent(Activation):
    def __init__(self, z):
        super().__init__()
        self.z = z

    @override_from_parent
    def activate(self, z):
        xp = self.backend
        z = xp.asarray(z, dtype=float)
        return xp.hiperbolic_tangent(z)

    @override_from_parent
    def derivative(self, z):
        xp = self.backend
        a = xp.asarray(a, dtype=float)
        return 1.0 - a**2

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
