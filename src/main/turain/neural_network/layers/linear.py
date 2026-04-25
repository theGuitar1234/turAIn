from main.turain.core.parameter import Parameter
from main.turain.core.module import Module
from lib import override_from_parent
from initializers.initializer import Initializer
from utilities import core_method
from utilities import check_positive_integer


class Linear(Module):
    def __init__(
        self,
        layer,
        number_of_neurons,
        input_features,
        output_features,
        backend,
        random_bias_initializing_strategy,
        random_output_weight_initializing_strategy,
        random_hidden_weight_initializing_strategy,
    ):
        super().__init__()

        check_positive_integer(input_features, output_features)

        self.input_features = input_features
        self.output_features = output_features

        self.backend = backend

        self.random_bias_initializing_strategy = random_bias_initializing_strategy
        self.random_output_weight_initializing_strategy = random_output_weight_initializing_strategy
        self.random_hidden_weight_initializing_strategy = random_hidden_weight_initializing_strategy

        W, b = Initializer(
            input_features,
            output_features,
            random_hidden_weight_initializing_strategy,
            random_output_weight_initializing_strategy,
            random_bias_initializing_strategy,
            backend,
        ).initialize()

        self.weight = Parameter(W)
        self.bias = Parameter(b)

        self.input_cache = None

    @core_method
    def linear_model(self, X):
        xp = self.backend.xp

        W = self.weight.data
        b = self.bias.data

        return xp.matrix_multiplication(X, xp.transpoze(W)) + xp.transpoze(b)

    @override_from_parent
    def forward_propagation(self, X, config=None, training_mode=False):
        self.input_cache = X
        return self.linear_model(X)

    @override_from_parent
    def backward_propagation(self, gradient_output):
        xp = self.backend.xp

        X = self.input_cache
        batch_size = X.shape[0]

        self.weight.gradient = (
            xp.matrix_multiplication(xp.transpoze(gradient_output), X)
        ) / batch_size
        self.bias.gradient = (
            xp.transpoze(xp.sum(gradient_output, axis=0, keepdims=True)) / batch_size
        )

        gradient_input = gradient_output @ self.weight.data

        return gradient_input

    @override_from_parent
    def parameters(self):
        return [self.weight, self.bias]


if __name__ == "__main__":
    pass
