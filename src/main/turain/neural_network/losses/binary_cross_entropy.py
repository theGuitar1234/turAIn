from main.turain.neural_network.losses.loss import Loss
from lib import override_from_parent
from utilities import TrainDefaults


class BinaryCrossEntropyLoss(Loss):
    def __init__(self, backend, cfg=None):
        super().__init__(backend, cfg)

    @override_from_parent
    def loss(self, true_label, prediction, size, xp):
        return (
            -xp.sum(true_label * xp.log(prediction) + (1.0 - true_label) * xp.log(1.0 - prediction))
            / size
        )

    @override_from_parent
    def loss_derivative(self, true_label, prediction, size):
        return (-(true_label / prediction) + ((1.0 - true_label) / (1.0 - prediction))) / size

    @override_from_parent
    def forward_propagation(self, prediction, true_label):
        xp = self.backend
        prediction = xp.clip(prediction, self.epsilon, 1 - self.epsilon)

        self.prediction_cache = prediction
        self.true_label_cache = true_label

        batch_size = true_label.shape[0]

        loss = self.loss(true_label, prediction, batch_size, xp)

        return loss

    @override_from_parent
    def backward_propagation(self):
        prediction = self.prediction_cache
        true_label = self.true_label_cache
        batch_size = true_label.shape[0]

        gradient_prediction = self.loss_derivative(true_label, prediction, batch_size)

        return gradient_prediction


if __name__ == "__main__":
    pass
