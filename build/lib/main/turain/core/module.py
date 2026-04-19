class Module:
    def __init__(self):
        self.training = True

    def forward_propagation(self, x):
        raise NotImplementedError

    def backward_propagation(self, gradient_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
