from main.turain.core.module import Module
from lib import override_from_parent


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    @override_from_parent
    def forward_propagation(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    @override_from_parent
    def backward_propagation(self, _gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward_propagation(_gradient)
        return gradient

    @override_from_parent
    def parameters(self):
        _parameters = []
        for layer in self.layers:
            _parameters.extend(layer.parameters())
        return _parameters

    def __repr__(self):
        return (
            f"NeuralNetwork("
            f"input_width={self.__input_width}, "
            f"layers={self.__layers}, "
            f"hidden_activation={self.__hidden_activation_type.name}, "
            f"output_activation={self.__output_activation_type.name}, "
            f"loss={self.__loss_type.name}, "
            f"parameters={self.parameter_count()})"
        )


if __name__ == "__main__":
    pass
