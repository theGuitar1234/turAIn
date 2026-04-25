from main.turain.neural_network.initializers import Initializer
from main.turain.core.parameter import Parameter
from main.turain.core.module import Module
from lib import override_from_parent
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
        xp = self.backend

        W = self.weight.data
        b = self.bias.data

        return xp.matrix_multiplication(X, xp.transpoze(W)) + xp.transpoze(b)

    @override_from_parent
    def forward_propagation(self, X, config=None, training_mode=False):
        self.input_cache = X
        return self.linear_model(X)

    @override_from_parent
    def backward_propagation(self, gradient_output):
        xp = self.backend

        X = self.input_cache
        batch_size = X.shape[0]

        W = self.weight
        B = self.bias

        w = W.data
        b = B.data

        dW = self.weight.gradient
        db = self.weight.gradient

        dW = (xp.matrix_multiplication(gradient_output.T, X)) / batch_size
        db = xp.transpoze(xp.sum(gradient_output, axis=0, keepdims=True).T) / batch_size

        gradient_input = gradient_output @ w

        return gradient_input

    @override_from_parent
    def parameters(self):
        return [self.weight, self.bias]


if __name__ == "__main__":
    pass
