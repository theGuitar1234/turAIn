from turain.backend.cpu import CPU
from turain.backend.gpu import GPU
from turain.models.sequential import Sequential
from turain.neural_network.activations.sigmoid import Sigmoid
from turain.neural_network.layers.linear import Linear
from turain.neural_network.activations.relu import ReLU
from turain.utilities.config import InitializationDefaults
from turain.utilities.enum import (
    BiasInititializationStrategy,
    HiddenActivationType,
    OutputActivationType,
    WeightInitializationStrategy,
)


backend = GPU()
xp = backend.xp

X = xp.random.standard_normal((6, 4)).astype(xp.float32)
Y = xp.random.standard_normal((6, 4)).astype(xp.float32)

number_of_classes = Y.shape[1]
number_of_features = X.shape[1]
print(f"number_of_features {number_of_features}")

layers = [number_of_features, number_of_classes]

init_layers = []
activation = ReLU(backend)
output_activation = Sigmoid(backend)
number_of_layers = len(layers)

initializer = InitializationDefaults(
    number_of_layers=number_of_layers,
    bias_initializing_strategy=BiasInititializationStrategy.ZERO,
    output_weight_initializing_strategy=WeightInitializationStrategy.ZERO,
    hidden_weight_initializing_strategy=WeightInitializationStrategy.ZERO,
    output_activation_type=OutputActivationType.SIGMOID,
    hidden_activation_type=HiddenActivationType.RELU,
)

for i in range(number_of_layers):
    if i != len(layers) - 1:
        linear = Linear(
            layer=i,
            number_of_features=number_of_features,
            output_features=number_of_classes,
            number_of_neurons=layers[i],
            backend=backend,
            initializer=initializer
        )
        init_layers.append(linear)
        init_layers.append(activation)
    linear = Linear(
        layer=i,
        number_of_features=number_of_features,
        output_features=number_of_classes,
        number_of_neurons=layers[i],
        backend=backend,
        initializer=initializer
    )
    init_layers.append(linear)
    init_layers.append(output_activation)

model = Sequential(init_layers)
model.move_to(backend)

for parameter in model.parameters():
    print(type(parameter.data), parameter.data.shape)