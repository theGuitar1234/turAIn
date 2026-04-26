from main.turain.backend.cpu import CPU
from main.turain.models.sequential import Sequential
from main.turain.neural_network.layers.linear import Linear
from main.turain.neural_network.activations.relu import ReLU
from main.turain.neural_network.losses.mean_squared_error import MeanSquaredErrorLoss
from main.turain.train.evaluator import Evaluator


def simple_weight_init(input_features, output_features, xp):
    return xp.random.standard_normal((output_features, input_features)) * 0.01


def simple_bias_init(output_features, xp):
    return xp.zeros((1, output_features))


backend = CPU()
xp = backend.xp

model = Sequential(
    Linear(4, 8, backend, simple_weight_init, simple_bias_init),
    ReLU(backend),
    Linear(8, 2, backend, simple_weight_init, simple_bias_init),
)

loss_function = MeanSquaredErrorLoss(backend)
evaluator = Evaluator(model, backend)

X = xp.random.standard_normal((6, 4)).astype(xp.float32)
Y = xp.random.standard_normal((6, 2)).astype(xp.float32)

result = evaluator.evaluate(X, Y, loss_function)

print(result["prediction"].shape)
print(result["loss"])
print(result["accuracy"])
