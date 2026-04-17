from typing import override
from src.main.python.core.module import Module


class Linear(Module):
    def __init__(self, input_features, output_features, backend, initializer):
        super().__init__()

    @override
    def forward_propagation(self, x):
        pass

    @override
    def backward_propagation(self, gradient_output):
        pass

    @override
    def parameters(self):
        pass
