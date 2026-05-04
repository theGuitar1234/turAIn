from turain.backend.cpu import CPU
from turain.models.sequential import Sequential
from turain.neural_network.activations.sigmoid import Sigmoid
from turain.neural_network.layers.linear import Linear
from turain.neural_network.activations.relu import ReLU
from turain.neural_network.losses.mean_squared_error import MeanSquaredErrorLoss
from turain.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from turain.train.evaluator import Evaluator
from turain.utilities.config import InitializationDefaults
from turain.utilities.enum import (
    BiasInititializationStrategy,
    HiddenActivationType,
    OutputActivationType,
    WeightInitializationStrategy,
)


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

loss_function = MeanSquaredErrorLoss(backend)
optimizer = StochasticGradientDescent(model.parameters(), backend, learning_rate=0.01)
# optimizer = Momentum(model.parameters(), learning_rate=0.01, beta=0.9, backend=backend)
# optimizer = Adam(model.parameters(), learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, backend=backend)

before = [parameter.data.copy() for parameter in model.parameters()]

prediction = model.forward_propagation(X)
loss = loss_function.forward_propagation(prediction, Y)

optimizer.zero_gradient()
gradient_loss = loss_function.backward_propagation()
model.backward_propagation(gradient_loss)
optimizer.step()

after = [parameter.data for parameter in model.parameters()]

print("loss:", loss)
for i, (b, a) in enumerate(zip(before, after)):
    changed = xp.any(b != a)
    print(f"parameter {i} changed:", bool(changed))
