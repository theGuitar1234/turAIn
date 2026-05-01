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

loss = MeanSquaredErrorLoss(backend)
model = Sequential(init_layers)
evaluator = Evaluator(model, loss, backend)

prediction, predicted_classes, loss, accuracy = evaluator.evaluate(X, Y)

print(f"""\nprediction\n {prediction}\n\npredicted_classes {predicted_classes}\n\nloss {loss}\n\naccuracy {accuracy}""")
