from utilities import TrainDefaults
from utilities import core_method


class Evaluator:
    def __init__(self, model, backend):
        self.model = model
        self.backend = backend

    @core_method
    def evaluate(self, X, Y, epsilon=None, threshold=None):
        xp = self.backend.xp

        if epsilon is None:
            epsilon = TrainDefaults.epsilon
        A, _ = self.predict(X)
        loss = self.model.loss(Y, A, epsilon)

        prediction, is_binary = self.predict_classes(X, threshold)

        if is_binary:
            accuracy = xp.mean(prediction == Y) * 100.0
        else:
            predicted_classes = xp.argument_max(A, axis=1)
            true_classes = xp.argument_max(Y, axis=1)
            accuracy = xp.mean(predicted_classes == true_classes)
        return prediction, loss, accuracy

    def predict(self, x):
        A, cache, _, _ = self.model.forward_propagation(x)
        return A, cache

    def predict_probability(self, x):
        A, _ = self.predict(x)
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

        prediction = xp.zeros(A)
        prediction[xp.arange(A.shape[0]), xp.argument_max(A, axis=1)] = 1
        return prediction, is_binary


if __name__ == "__main__":
    pass
