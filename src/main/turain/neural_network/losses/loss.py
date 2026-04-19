from utilities import TrainDefaults


class Loss:
    def __init__(self, backend, epsilon=None):
        self.backend = backend
        if epsilon is None:
            self.epsilon = TrainDefaults().epsilon

        self.prediction_cache = None
        self.true_label_cache = None

    def forward_propagation(self, prediction, true_label):
        raise NotImplementedError

    def backward_propagation(self):
        raise NotImplementedError
