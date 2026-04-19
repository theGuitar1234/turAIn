from utilities import TrainDefaults


class Optimizer:
    def __init__(self, parameters, learning_rate=None):
        self.parameters = list(parameters)
        if learning_rate is None:
            self.learning_rate = TrainDefaults.learning_rate

    def step(self):
        raise NotImplementedError

    def zero_gradient(self):
        for p in self.parameters:
            p.gradient = None
