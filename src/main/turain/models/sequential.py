from main.turain.core.module import Module


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward_propagation(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    def backward_propagation(self, _gradient):
        for layer in range(len(self.layers) - 1, -1, -1):
            gradient = layer.backward_propagation(_gradient)
        return gradient

    def parameters(self):
        _parameters = []
        for layer in self.layers:
            _parameters.extend(layer.parameters())
        return _parameters