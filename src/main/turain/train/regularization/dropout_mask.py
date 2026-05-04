from ...core import Module
from ...utilities import TrainDefaults
from ...lib import override_from_parent


class DropoutMask(Module):
    def __init__(self, drop_out_rate=None, backend=None, rate=None):
        super().__init__()

        if backend is None and hasattr(drop_out_rate, "xp"):
            backend = drop_out_rate
            drop_out_rate = rate

        if drop_out_rate is None:
            drop_out_rate = TrainDefaults.drop_out_rate

        if backend is None:
            raise ValueError("DropoutMask requires a backend.")
        if not 0.0 <= drop_out_rate < 1.0:
            raise ValueError("drop_out_rate must be in the range [0.0, 1.0).")

        self.drop_out_rate = drop_out_rate
        self.keep_rate = 1.0 - self.drop_out_rate
        self.mask = None
        self.backend = backend

    @override_from_parent
    def forward_propagation(self, x):
        if not self.training or self.drop_out_rate == 0.0:
            self.mask = None
            return x

        self.mask = self.bernoulli(x.shape, x.dtype)
        return (x * self.mask) / self.keep_rate

    @override_from_parent
    def backward_propagation(self, gradient_output):
        if not self.training or self.mask is None or self.drop_out_rate == 0.0:
            return gradient_output

        return (gradient_output * self.mask) / self.keep_rate

    @override_from_parent
    def parameters(self):
        return []

    def bernoulli(self, shape, dtype=float):
        xp = self.backend.xp
        return (xp.random.random(shape) < self.keep_rate).astype(dtype)
