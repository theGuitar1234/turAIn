from turain.backend.cpu import CPU
from turain.models.sequential import Sequential
from turain.neural_network.activations.sigmoid import Sigmoid
from turain.neural_network.layers.linear import Linear
from turain.neural_network.activations.relu import ReLU
from turain.neural_network.losses.mean_squared_error import MeanSquaredErrorLoss
from turain.train.evaluator import Evaluator
from turain.utilities.config import InitializationDefaults
from turain.utilities.enum import (
    BiasInititializationStrategy,
    HiddenActivationType,
    OutputActivationType,
    WeightInitializationStrategy,
)

from turain.neural_network.initializers.bias_initializer import BiasInitialzer
from turain.neural_network.initializers.weight_initializer import WeightInitializer

backend = CPU()
xp = backend.xp

number_of_classes = 2

X = xp.random.standard_normal((6, 4)).astype(xp.float32)
Y = xp.random.standard_normal((6, number_of_classes)).astype(xp.float32)

number_of_features = X.shape[1]
print(f"number_of_features {number_of_features}")

layers = [5, 2]

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

loss_function = MeanSquaredErrorLoss(backend)
model = Sequential(init_layers, loss_function)
evaluator = Evaluator(model, backend)

result = evaluator.evaluate(X, Y)

print(result["prediction"].shape)
print(result["loss"])
print(result["accuracy"])
