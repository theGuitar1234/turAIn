from main.turain.models.sequential import Sequential
from main.turain.neural_network.activations.hiperbolic_tangent import HiperbolicTangent
from main.turain.neural_network.activations.leaky_relu import LeakyReLU
from main.turain.neural_network.activations.relu import ReLU
from main.turain.neural_network.activations.sigmoid import Sigmoid
from main.turain.neural_network.activations.softmax import Softmax
from main.turain.neural_network.layers.linear import Linear
from main.turain.train.regularization.dropout_mask import DropoutMask
from main.turain.utilities.config import TrainDefaults
from main.turain.utilities.enum import HiddenActivationType, OutputActivationType


class ModelFactory:

    @staticmethod
    def build_mlp(
        layer_sizes,
        backend,
        weight_initializer,
        bias_initializer,
        output_bias_initializer,
        hidden_bias_initializer,
        output_weight_initializer,
        hidden_weight_initializer,
        hidden_activation=None,
        output_activation=None,
        config=None,
    ):
        if config is None:
            config = TrainDefaults()
        drop_out_rate = config.drop_out_rate

        layers = []
        number_of_linear_layers = len(layer_sizes) - 1

        for index in range(number_of_linear_layers):
            in_features = layer_sizes[index]
            out_features = layer_sizes[index + 1]

            is_output_layer = index == number_of_linear_layers - 1

            weight_initializer = (
                output_weight_initializer if is_output_layer else hidden_weight_initializer
            )
            bias_initializer = (
                output_bias_initializer if is_output_layer else hidden_bias_initializer
            )

            layers.append(
                Linear(
                    in_features,
                    out_features,
                    backend,
                    weight_initializer,
                    bias_initializer,
                )
            )

            if not is_output_layer and hidden_activation is not None:
                match hidden_activation:
                    case HiddenActivationType.RELU:
                        layers.append(ReLU(backend))
                    case HiddenActivationType.LEAKY_RELU:
                        layers.append(LeakyReLU(backend))
                    case HiddenActivationType.SIGMOID:
                        layers.append(Sigmoid(backend))
                    case HiddenActivationType.TANH:
                        layers.append(HiperbolicTangent(backend))
                    case _:
                        raise ValueError(
                            f"Unknown hidden activation, supported values are {list(HiddenActivationType)}"
                        )
                if drop_out_rate > 0.0:
                    layers.append(DropoutMask(backend, rate=drop_out_rate))
            if is_output_layer and output_activation is not None:
                match output_activation:
                    case OutputActivationType.SIGMOID:
                        layers.append(Sigmoid(backend))
                    case OutputActivationType.SOFTMAX:
                        layers.append(Softmax(backend))
                    case _:
                        raise ValueError(
                            f"Unknown output activation, supported values are {list(OutputActivationType)}"
                        )
        return Sequential(*layers)
