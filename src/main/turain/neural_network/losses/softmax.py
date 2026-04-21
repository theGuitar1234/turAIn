from main.turain.neural_network.losses.loss import Loss
from lib import override_from_parent


class SoftmaxLoss(Loss):
    def __init__(self, backend, cfg=None):
        super().__init__(backend, cfg)

    @override_from_parent
    def forward_propagation(self, z, true_label):
        xp = self.backend

        z_shifted = z - xp.max(z, axis=1, keepdims=True)
        e_to_the_power_z = xp.e_to_the_power(z_shifted)
        probabilities = e_to_the_power_z / xp.sum(e_to_the_power_z, axis=1, keepdims=True)
        probabilities = xp.clip(probabilities, self.epsilon, 1.0)

        self.prediction_cache = probabilities
        self.true_label_cache = self.true_label_cache

        batch_size = true_label.shape[0]
        loss = -xp.sum(true_label * xp.log(probabilities)) / batch_size

        return loss

    @override_from_parent
    def backward_propagation(self):
        probabilities = self.true_label_cache
        true_label = self.true_label_cache
        batch_size = true_label.shape[0]

        return self.output_delta(probabilities, true_label) / batch_size

    def output_delta(prediction, true_label):
        return prediction - true_label


if __name__ == "__main__":
    pass
