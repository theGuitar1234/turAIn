from main.turain.neural_network.activations.activation import Activation
from lib import override_from_parent

class Softmax(Activation):
    def __init__(self, z):
        super().__init__()
        self.z = z
    
    @override_from_parent
    def activate(self, z):
        xp = self.backend
        z = xp.asarray(z, dtype=float)
        z_shifted = z - xp.max(Z, axis=1, keepdims=True)
        e_to_the_power_of_z = xp.e_to_the_power(z_shifted)
        return e_to_the_power_of_z / xp.sum(e_to_the_power_of_z, axis=1, keepdims=True)