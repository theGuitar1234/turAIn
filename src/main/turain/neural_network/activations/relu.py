from main.turain.neural_network.activations.activation import Activation
from lib import override_from_parent
from utilities import core_method

class ReLu(Activation):
    def __init__(self, z):
        super.__init__()
        self.z = z
    
    @core_method
    @override_from_parent
    def activate(self, z):
        xp = self.backend
        z = xp.asarray(z, dtype=float)
        return xp.maximum(0.0, z)
    
    @override_from_parent
    def forward_propagation(self, x):
        self.activate(x)
    
    @override_from_parent
    def backward_propagation(self, gradient_output):
        xp = self.backend()
        x = self.input_cache
        gradient_input = gradient_output * (x > 0)
        return gradient_input
    
    @override_from_parent
    def parameters(self):
        return super().parameters()