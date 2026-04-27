from .activation import Activation
from ...lib import override_from_parent
from ...utilities import core_method


class ReLU(Activation):
    def __init__(self, backend):
        super().__init__(backend)

    @override_from_parent
    def activate(self, z):
        xp = self.backend.xp
        z = xp.asarray(z, dtype=float)
        return xp.maximum(0.0, z)

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


    
