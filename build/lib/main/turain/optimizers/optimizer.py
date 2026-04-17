class Optimizer:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_gradient(self):
        for p in self.parameters:
            p.gradient = None
