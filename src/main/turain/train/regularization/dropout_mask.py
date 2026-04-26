from ...core import Module
from ...utilities import TrainDefaults
from ...lib import override_from_parent

class DropoutMask(Module):
    def __init__(self, backend, config=None):
        if config is None:
            config = TrainDefaults()
        super().__init__()
        self.drop_out_rate = config.drop_out_rate
        self.keep_rate = 1 - self.drop_out_rate
        self.mask = None
        self.backend = backend

    @override_from_parent
    def forward_propagation(self, x):
        xp = self.backend.xp
        if not self.training or self.drop_out_rate == 0.0:
            self.mask = None
            return x
        self.mask = self.bernoulli(x.shape)
        return (x * self.mask) / self.keep_rate

    @override_from_parent
    def backward_propagation(self, gradient_output):
        if not self.training or self.mask is None or self.drop_out_rate == 0.0:
            return gradient_output
        return (gradient_output * self.mask) / self.keep_rate

    @override_from_parent
    def parameters(self):
        return []

    def bernoulli(self, shape):
        xp = self.backend.xp
        return (xp.random.random(shape) < self.keep_rate).astype(float)



    
