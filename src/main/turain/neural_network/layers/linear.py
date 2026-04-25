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

    # def _initialize_weight(self, initializer, xp, input_features, output_features):
    #     if callable(initializer):
    #         weight = initializer(input_features, output_features, xp)
    #     elif hasattr(initializer, "initialize"):
    #         weight = initializer.initialize(input_features, output_features, xp)
    #     else:
    #         raise TypeError("weight_initializer must be callable or have initialize(...)")

    #     weight = xp.asarray(weight, dtype=xp.float32)

    #     expected_shape = (output_features, input_features)
    #     if weight.shape != expected_shape:
    #         raise ValueError(f"Weight shape must be {expected_shape}, got {weight.shape}")

    #     return weight

    # def _initialize_bias(self, initializer, xp, output_features):
    #     if callable(initializer):
    #         bias = initializer(output_features, xp)
    #     elif hasattr(initializer, "initialize"):
    #         bias = initializer.initialize(output_features, xp)
    #     else:
    #         raise TypeError("bias_initializer must be callable or have initialize(...)")

    #     bias = xp.asarray(bias, dtype=xp.float32)

    #     if bias.ndim == 1:
    #         bias = bias.reshape(1, -1)
    #     elif bias.shape == (output_features, 1):
    #         bias = bias.T

    #     expected_shape = (1, output_features)
    #     if bias.shape != expected_shape:
    #         raise ValueError(f"Bias shape must be {expected_shape}, got {bias.shape}")

    #     return bias

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
