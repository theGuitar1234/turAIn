from .activation import Activation
from lib import override_from_parent
from utilities import core_method
from utilities import TrainDefaults


class LeakyReLu(Activation):
    def __init__(self, z, _negative_slope=None):
        super().__init__()

        if _negative_slope is None:
            _negative_slope = TrainDefaults.negative_slope

        self.negative_slope = _negative_slope
        self.input_cache = None
        self.z = z

    @override_from_parent
    def activate(self, z):
        xp = self.backend.xp
        z = xp.asarray(z, dtype=float)
        return xp.where(z > 0, z, self.negative_slope * z)

    @override_from_parent
    def derivative(self, z):
        return self.activate(z)

    @override_from_parent
    def forward_propagation(self, x):
        self.input_cache = x
        return self.activate(x)

    @override_from_parent
    def backward_propagation(self, gradient_output):
        x = self.input_cache
        gradient_input = gradient_output * self.derivative(x)
        return gradient_input



    
