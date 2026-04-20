from main.turain.neural_network.losses.loss import Loss
from lib import override_from_parent


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, backend):
        self.xp = backend
    
    @override_from_parent
    def forward_propagation(self, prediction, true_label):
        return super().forward_propagation(prediction, true_label)
    
    @override_from_parent
    def backward_propagation(self):
        return super().backward_propagation()
