
from turain.utilities.annotation import core_method
from turain.utilities.config import TrainDefaults

class Evaluator:
    def __init__(self, model, loss, backend):
        self.model = model
        self.loss = loss
        self.backend = backend

    @core_method
    def evaluate(self, X, Y, config=None):
        xp = self.backend.xp

        if config is None:
            config = TrainDefaults()

        epsilon = config.epsilon
        threshold = config.threshold

        # A, _ = self.predict(X)
        A = self.predict(X)
        loss = self.loss.loss(A, Y, Y.shape[0])

        prediction, is_binary = self.predict_classes(X, threshold)

        if is_binary:
            accuracy = xp.mean(prediction == Y) * 100.0
        else:
            predicted_classes = xp.argmax(A, axis=1)
            true_classes = xp.argmax(Y, axis=1)
        accuracy = xp.mean(predicted_classes == true_classes) * 100.0

        return prediction, predicted_classes, loss, self.backend.to_cpu(accuracy)

    def predict(self, x):
        # A, cache, _, _ = self.model.forward_propagation(x)
        A = self.model.forward_propagation(x)
        return A

    def predict_probability(self, x):
        # A, _ = self.predict(x)
        A = self.predict(x)
        return A

    def predict_classes(self, x, threshold=None):
        xp = self.backend.xp

        if threshold is None:
            threshold = TrainDefaults.threshold

        A = self.predict_probability(x)

        is_binary = A.shape[1] == 1
        if is_binary:
            prediction = (A >= threshold).astype(float)
            return prediction, is_binary

        prediction = xp.zeros_like(A)
        prediction[xp.arange(A.shape[0]), xp.argmax(A, axis=1)] = 1
        return prediction, is_binary
