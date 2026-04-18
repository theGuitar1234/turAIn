from main.turain.core.module import Module


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward_propagation(self, x):
        raise NotImplementedError

    def backward_propagation(self, gradient):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError
