from turain.backend.cpu import CPU
from turain.models.sequential import Sequential
from turain.neural_network.layers.linear import Linear
from turain.neural_network.activations.relu import ReLU
from turain.neural_network.losses.mean_squared_error import MeanSquaredErrorLoss
from turain.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from turain.train.train import Train
from turain.utilities.config import TrainDefaults
from turain.data.batch_loader import BatchLoader


def simple_weight_init(input_features, output_features, xp):
    return xp.random.standard_normal((output_features, input_features)) * 0.01


def simple_bias_init(output_features, xp):
    return xp.zeros((1, output_features))


backend = CPU()
xp = backend.xp

X_train = xp.random.standard_normal((20, 4)).astype(xp.float32)
Y_train = xp.random.standard_normal((20, 2)).astype(xp.float32)

X_valid = xp.random.standard_normal((8, 4)).astype(xp.float32)
Y_valid = xp.random.standard_normal((8, 2)).astype(xp.float32)

train_loader = BatchLoader(X_train, Y_train, batch_size=4, backend=backend, shuffle=True)
valid_loader = BatchLoader(X_valid, Y_valid, batch_size=4, backend=backend, shuffle=False)

model = Sequential(
    Linear(4, 8, backend, simple_weight_init, simple_bias_init),
    ReLU(backend),
    Linear(8, 2, backend, simple_weight_init, simple_bias_init),
)

loss_function = MeanSquaredErrorLoss(backend)
optimizer = StochasticGradientDescent(model.parameters(), learning_rate=0.01)

config = TrainDefaults()
config.epochs = 3
config.threshold = 0.5

trainer = Train(
    model=model,
    loss_function=loss_function,
    optimizer=optimizer,
    config=config,
)

results = trainer.fit(train_loader, valid_loader)

print(results.train_losses)
print(results.validation_losses)
