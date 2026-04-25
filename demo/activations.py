from main.turain.backend.cpu import CPU
from main.turain.neural_network.activations.relu import ReLU
from main.turain.neural_network.activations.leaky_relu import LeakyReLU
from main.turain.neural_network.activations.sigmoid import Sigmoid
from main.turain.neural_network.activations.hiperbolic_tangent import HiperbolicTangent

backend = CPU()
xp = backend.xp

x = xp.asarray(
    [
        [-2.0, -0.5, 0.0, 1.0, 3.0],
        [1.5, -1.2, 0.2, 0.0, -4.0],
    ],
    dtype=xp.float32,
)

gradient_output = xp.ones_like(x)

activations = [
    ReLU(backend),
    LeakyReLU(backend, alpha=0.01),
    Sigmoid(backend),
    HiperbolicTangent(backend),
]

for activation in activations:
    y = activation.forward_propagation(x)
    dx = activation.backward_propagation(gradient_output)

    print(activation.__class__.__name__)
    print(" forward shape:", y.shape)
    print(" backward shape:", dx.shape)
    print()
