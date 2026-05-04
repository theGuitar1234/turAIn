from turain.backend.cpu import CPU
from turain.models.sequential import Sequential
from turain.neural_network.activations.sigmoid import Sigmoid
from turain.neural_network.layers.linear import Linear
from turain.neural_network.activations.relu import ReLU
from turain.neural_network.losses.mean_squared_error import MeanSquaredErrorLoss
from turain.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from turain.train.evaluator import Evaluator
from turain.train.finalizer import Finalizer
from turain.train.logger import Logger
from turain.train.plotter import Plotter
from turain.train.regularization.dropout_mask import DropoutMask
from turain.train.regularization.l2_regularization import L2Regularization
from turain.train.schedulers.exponential_decay import ExponentialDecay
from turain.train.state_tracker import StateTracker
from turain.train.train import Train
from turain.utilities.config import InitializationDefaults, TrainDefaults
from turain.utilities.enum import (
    BiasInititializationStrategy,
    HiddenActivationType,
    OutputActivationType,
    WeightInitializationStrategy,
)
from turain.data.batch_loader import BatchLoader

backend = CPU()
xp = backend.xp

X_train = xp.random.standard_normal((20, 4)).astype(xp.float32)
Y_train = xp.random.standard_normal((20, 2)).astype(xp.float32)

X_valid = xp.random.standard_normal((8, 4)).astype(xp.float32)
Y_valid = xp.random.standard_normal((8, 2)).astype(xp.float32)

number_of_classes = Y_train.shape[1]
number_of_features = X_train.shape[1]
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

train_loader = BatchLoader(X_train, Y_train, batch_size=4, backend=backend, shuffle=True)
validation_loader = BatchLoader(X_valid, Y_valid, batch_size=4, backend=backend, shuffle=False)

model = Sequential(
    [
        Linear(
            0, 4, number_of_features, number_of_classes, backend=backend, initializer=initializer
        ),
        activation,
        Linear(
            1, number_of_classes, 4, number_of_classes, backend=backend, initializer=initializer
        ),
        output_activation,
    ]
)

loss_function = MeanSquaredErrorLoss(backend)
optimizer = StochasticGradientDescent(model.parameters(), backend, learning_rate=0.01)

config = TrainDefaults()
config.epochs = 10
config.threshold = 0.5
config.step_size = 100

evaluator = Evaluator(model, loss_function, backend)
finalizer = Finalizer(evaluator, backend, number_of_classes)

trainer = Train(
    model=model,
    loss_function=loss_function,
    optimizer=optimizer,
    state_tracker=StateTracker(),
    logger=Logger(),
    plotter=Plotter(),
    finalizer=finalizer,
    scheduler=ExponentialDecay(config.learning_rate),
    l2_regularizer=L2Regularization(config.l2_lambda, backend),
    l1_regularizer=DropoutMask(config.drop_out_rate, backend),
)

results = trainer.fit(
    X_train=X_train,
    Y_train=Y_train,
    X_valid=X_valid,
    Y_valid=Y_valid,
    train_loader=train_loader,
    validation_loader=validation_loader,
    config=config,
    track_state=True,
    finalize=True,
    log_predictions=True,
    plot=True,
    plot_real_time=True,
)

# print(results)
