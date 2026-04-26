from main.turain.backend.cpu import CPU
from main.turain.models.sequential import Sequential
from main.turain.neural_network.layers.linear import Linear
from main.turain.neural_network.activations.relu import ReLU


def simple_weight_init(input_features, output_features, xp):
    return xp.random.standard_normal((output_features, input_features)) * 0.01


def simple_bias_init(output_features, xp):
    return xp.zeros((1, output_features))


cpu = CPU()

model = Sequential(
    Linear(4, 8, cpu, simple_weight_init, simple_bias_init),
    ReLU(cpu),
    Linear(8, 2, cpu, simple_weight_init, simple_bias_init),
)

model.move_to(cpu)

for parameter in model.parameters():
    print(type(parameter.data), parameter.data.shape)