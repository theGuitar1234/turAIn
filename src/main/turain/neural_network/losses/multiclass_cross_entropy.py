from main.turain.neural_network.losses.loss import Loss
from lib import override_from_parent


class MultiClassCrossEntropyLoss(Loss):
    def __init__(self, backend, epsilon=None):
        super().__init__(backend, epsilon)

    @override_from_parent
    def forward_propagation(self, prediction, true_label):
        xp = self.backend

        prediction = xp.clip(
            prediction,
            self.epsilon,
            1.0,
        )

        self.prediction_cache = prediction
        self.true_label_cache = true_label

        batch_size = true_label.shape[0]

        loss = -xp.sum(true_label * xp.log(prediction)) / batch_size
        return loss

    @override_from_parent
    def backward_propagation(self):
        prediction = self.prediction_cache
        true_label = self.true_label_cache
        batch_size = true_label.shape[0]

        gradient_prediction = -(true_label / prediction) / batch_size
        return gradient_prediction
