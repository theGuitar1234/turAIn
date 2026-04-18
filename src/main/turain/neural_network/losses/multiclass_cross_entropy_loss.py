from main.turain.neural_network.losses.loss import Loss
from lib import override_from_parent

class MultiClassCrossEntropyLoss(Loss):
    def __init__(self, backend, epsilon=1e-12):
        raise NotImplementedError

    @override_from_parent
    def forward_propagation():
        raise NotImplementedError

    @override_from_parent
    def backward_propagation(self):
        raise NotImplementedError
