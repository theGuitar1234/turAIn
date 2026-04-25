from main.turain.backend.cpu import CPU
from main.turain.models.sequential import Sequential
from main.turain.neural_network.layers.linear import Linear


def simple_weight_init(input_features, output_features, xp):
    return xp.random.standard_normal((output_features, input_features)) * 0.01


def simple_bias_init(output_features, xp):
    return xp.zeros((1, output_features))


backend = CPU()
model = Sequential(
    Linear(4, 8, backend, simple_weight_init, simple_bias_init),
    Linear(8, 2, backend, simple_weight_init, simple_bias_init),
)
x = backend.xp.random.standard_normal((5, 4)).astype(backend.xp.float32)
gradient_output = backend.xp.random.standard_normal((5, 2)).astype(backend.xp.float32)
output = model.forward_propagation(x)
gradient_input = model.backward_propagation(gradient_output)
print("output shape:", output.shape)
print("gradient_input shape:", gradient_input.shape)
for parameter in model.parameters():
    print(parameter.role, parameter.data.shape, parameter.gradient.shape)
