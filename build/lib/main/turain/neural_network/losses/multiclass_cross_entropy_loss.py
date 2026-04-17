from typing import override

from src.main.python.neural_network.losses.loss import Loss


class MultiClassCrossEntropyLoss(Loss):
    def __init__(self, backend, epsilon=1e-12):
        raise NotImplementedError

    @override
    def forward_propagation():
        raise NotImplementedError

    @override
    def backward_propagation(self):
        raise NotImplementedError
