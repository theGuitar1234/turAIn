from utilities import TrainDefaults
from utilities import core_method


class Loss:
    def __init__(self, backend, cfg=None):
        if cfg is None:
            cfg = TrainDefaults()
        self.epsilon = cfg.epsilon
        self.backend = backend

        self.prediction_cache = None
        self.true_label_cache = None

    @core_method
    def loss(self):
        raise NotImplementedError
    
    def loss_derivative(self):
        raise NotImplementedError

    def forward_propagation(self):
        raise NotImplementedError

    def backward_propagation(self):
        raise NotImplementedError
