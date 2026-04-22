from main.turain.models.sequential import Sequential
from main.turain.neural_network.activations.relu import ReLu
from main.turain.neural_network.layers.linear import Linear
from main.turain.train.regularization.dropout_mask import DropoutMask


class ModelFactory:
    @staticmethod
    def build_mlp(layer_sizes, backend, weight_initializer, bias_initializer, hidden_activation="relu", dropout_probability=0.0):
        layers = []

        for index in range(len(layer_sizes) - 1):
            in_features = layer_sizes[index]
            out_features = layer_sizes[index + 1]

            layers.append(
                Linear(
                    in_features,
                    out_features,
                    backend,
                    weight_initializer,
                    bias_initializer,
                )
            )

            is_output_layer = index == len(layer_sizes) - 2
            if not is_output_layer:
                if hidden_activation == "relu":
                    layers.append(ReLu(backend))

                if dropout_probability > 0.0:
                    layers.append(DropoutMask(backend, probability=dropout_probability))

        return Sequential(*layers)