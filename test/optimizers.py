from turain.backend.cpu import CPU
from turain.models.sequential import Sequential
from turain.neural_network.layers.linear import Linear
from turain.neural_network.activations.relu import ReLU
from turain.neural_network.losses.mean_squared_error import MeanSquaredErrorLoss
from turain.optimizers.stochastic_gradient_descent import StochasticGradientDescent


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
optimizer = StochasticGradientDescent(model.parameters(), learning_rate=0.01)
# optimizer = Momentum(model.parameters(), learning_rate=0.01, beta=0.9, backend=backend)
# optimizer = Adam(model.parameters(), learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, backend=backend)

x = xp.random.standard_normal((5, 4)).astype(xp.float32)
y = xp.random.standard_normal((5, 2)).astype(xp.float32)

before = [parameter.data.copy() for parameter in model.parameters()]

prediction = model.forward_propagation(x)
loss = loss_function.forward_propagation(prediction, y)

optimizer.zero_gradient()
gradient_loss = loss_function.backward_propagation()
model.backward_propagation(gradient_loss)
optimizer.step()

after = [parameter.data for parameter in model.parameters()]

print("loss:", loss)
for i, (b, a) in enumerate(zip(before, after)):
    changed = xp.any(b != a)
    print(f"parameter {i} changed:", bool(changed))
