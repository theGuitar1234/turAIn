from .loss import Loss
from lib import override_from_parent


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, backend, config=None):
        super().__init__(backend, config)

    @override_from_parent
    def forward_propagation(self, z, true_label):
        xp = self.backend.xp

        z_shifted = z - xp.max(z, axis=1, keepdims=True)
        exp_z = xp.exp(z_shifted)
        probabilities = exp_z / xp.sum(exp_z, axis=1, keepdims=True)
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



    
