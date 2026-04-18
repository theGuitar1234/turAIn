from main.turain.neural_network.activations.activation import Activation
from lib import override_from_parent
from utilities import core_method
from utilities import TrainDefaults

class LeakyReLu(Activation):
    def __init__(self, z, _negative_slope=None):
        super.__init__()
        
        if _negative_slope is None:
            _negative_slope = TrainDefaults.negative_slope
        self.negative_slope = _negative_slope
        self.input_cache = None
        self.z = z
    
    @core_method
    @override_from_parent
    def activate(self, z):
        xp = self.backend
        z = xp.asarray(z, dtype=float)
        return xp.where(z > 0, z, self.negative_slope * z)
    
    @override_from_parent
    def forward_propagation(self, x):
        self.input_cache = x
        return self.activate(x)
    
    @override_from_parent
    def backward_propagation(self, gradient_output):
        xp = self.backend
        
        x = self.input_cache
        local_gradient = self.activate(x)
        gradient_input = gradient_output * local_gradient
        return gradient_input
    
    @override_from_parent
    def parameters(self):
        return super().parameters()