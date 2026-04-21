from main.turain.neural_network.losses.loss import Loss
from lib import override_from_parent


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, backend, cfg=None):
        super().__init__(backend, cfg)

    @override_from_parent
    def loss(self, prediction, true_label, xp, size):
        return xp.sum((prediction - true_label) * (prediction - true_label)) / size

    @override_from_parent
    def loss_derivative(self, prediction, true_label):
        return prediction - true_label

    @override_from_parent
    def forward_propagation(self, prediction, true_label):
        xp = self.backend

        self.prediction_cache = prediction
        self.true_label_cache = true_label

        batch_size = true_label.shape[0]
        loss = self.loss(prediction, true_label, xp, batch_size)

        return loss

    @override_from_parent
    def backward_propagation(self):
        prediction = self.prediction_cache
        true_label = self.true_label_cache
        batch_size = true_label.shape[0]

        return (2.0 * (prediction - true_label)) / batch_size
