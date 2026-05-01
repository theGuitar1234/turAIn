from turain.core.parameter import Parameter
from turain.core.module import Module
from turain.neural_network.initializers.bias_initializer import BiasInitialzer
from turain.neural_network.initializers.weight_initializer import WeightInitializer
from turain.utilities.config import InitializationDefaults, TrainDefaults

from ...utilities import core_method
from ...utilities import check_positive_integer
from ...lib import override_from_parent


class Linear(Module):
    def __init__(
        self,
        layer,
        number_of_neurons,
        number_of_features,
        output_features,
        backend,
        initializer=None,
        config=None,
    ):
        super().__init__()

        if initializer is None:
            raise ValueError(
                f"Initializer is unset, consider using {InitializationDefaults.__name__}"
            )
        bias_initializing_strategy = initializer.bias_initializing_strategy
        output_weight_initializing_strategy = initializer.output_weight_initializing_strategy
        hidden_weight_initializing_strategy = initializer.hidden_weight_initializing_strategy

        if config is None:
            config = TrainDefaults()

        check_positive_integer(number_of_features, output_features)

        self.input_features = number_of_features
        self.output_features = output_features

        self.backend = backend

        self.bias_initializing_strategy = bias_initializing_strategy
        self.output_weight_initializing_strategy = output_weight_initializing_strategy
        self.hidden_weight_initializing_strategy = hidden_weight_initializing_strategy

        output_activation_type = initializer.output_activation_type
        hidden_activation_type = initializer.hidden_activation_type
        number_of_layers = initializer.number_of_layers

        is_output_layer = layer == number_of_layers - 1

        weight_initializer = WeightInitializer(
            layer=layer,
            is_output_layer=is_output_layer,
            backend=backend,
            number_of_neurons=number_of_neurons,
            number_of_features=number_of_features,
            output_features=output_features,
            hidden_weight_initializing_strategy=hidden_weight_initializing_strategy,
            output_weight_initializing_strategy=output_weight_initializing_strategy,
            bias_initializing_strategy=bias_initializing_strategy,
            output_activation_type=output_activation_type,
            hidden_activation_type=hidden_activation_type,
        )
        W = weight_initializer.initialize()

        bias_initializer = BiasInitialzer(
            is_output_layer=is_output_layer,
            number_of_neurons=number_of_neurons,
            backend=backend,
            bias_initializing_strategy=bias_initializing_strategy,
            output_activation_type=output_activation_type,
            hidden_activation_type=hidden_activation_type,
        )
        b = bias_initializer.initialize(config)

        self.weight = Parameter(W)
        self.bias = Parameter(b)

        self.input_cache = None

    @core_method
    def linear_model(self, X):
        W = self.weight.data
        b = self.bias.data
        return X @ W.T + b.T

    @override_from_parent
    def forward_propagation(self, X, config=None, training_mode=False):
        self.input_cache = X
        return self.linear_model(X)

    @override_from_parent
    def backward_propagation(self, gradient_output):
        xp = self.backend.xp

        X = self.input_cache
        batch_size = X.shape[0]

        self.weight.gradient = (gradient_input.T @ X) / batch_size
        self.bias.gradient = xp.sum(gradient_output, axis=0, keepdims=True).T / batch_size

        gradient_input = gradient_output @ self.weight.data

        return gradient_input

    @override_from_parent
    def parameters(self):
        return [self.weight, self.bias]
