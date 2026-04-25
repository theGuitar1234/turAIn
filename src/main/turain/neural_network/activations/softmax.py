from main.turain.neural_network.activations.activation import Activation
from lib import override_from_parent


class Softmax(Activation):
    def __init__(self, z):
        super().__init__()
        self.z = z

    @override_from_parent
    def activate(self, z):
        xp = self.backend.xp
        z = xp.asarray(z, dtype=float)
        z_shifted = z - xp.max(z, axis=1, keepdims=True)
        exp_of_z = xp.exp(z_shifted)
        return exp_of_z / xp.sum(exp_of_z, axis=1, keepdims=True)


if __name__ == "__main__":
    pass
