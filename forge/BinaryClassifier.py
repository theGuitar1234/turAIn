from turain.runtime.backend_selector import BackendSelector
from turain.utilities import Device

from turain.data.deserialization import DeSerialization
from turain.data.dataset_preparer import DatasetPreparer
from turain.data.batch_loader import BatchLoader

from turain.models.model_factory import ModelFactory

from turain.neural_network.initializers.layer_initializer import LayerInitializer
from turain.neural_network.losses.binary_cross_entropy import BinaryCrossEntropyLoss

from turain.optimizers.stochastic_gradient_descent import StochasticGradientDescent

from turain.train.train import Train
from turain.utilities import TrainDefaults
from turain.train.evaluator import Evaluator
from turain.train.finalizer import Finalizer
from turain.train.early_stopping import EarlyStopping
from turain.train.state_tracker import BestStateTracker

from turain.metrics.confusion_matrix import ConfusionMatrix
from turain.metrics.error_analysis import ErrorAnalysis
from turain.metrics.prediction_logger import PredictionLogger


backend = BackendSelector.select(Device.CPU)

dataset = DeSerialization.load_from_npz("data/npz/dataset.npz")
prepared = DatasetPreparer.prepare(
    X_train=dataset["X_train"],
    Y_train=dataset["Y_train"],
    X_valid=dataset["X_valid"],
    Y_valid=dataset["Y_valid"],
    X_test=dataset["X_test"],
    Y_test=dataset["Y_test"],
    flatten_features=False,
    one_hot=False,
    cast_labels_to_int=True,
)

X_train = backend.xp.asarray(prepared["X_train"], dtype=backend.xp.float32)
Y_train = backend.xp.asarray(prepared["Y_train"].reshape(-1, 1), dtype=backend.xp.float32)

X_valid = backend.xp.asarray(prepared["X_valid"], dtype=backend.xp.float32)
Y_valid = backend.xp.asarray(prepared["Y_valid"].reshape(-1, 1), dtype=backend.xp.float32)

X_test = backend.xp.asarray(prepared["X_test"], dtype=backend.xp.float32)
Y_test = backend.xp.asarray(prepared["Y_test"].reshape(-1, 1), dtype=backend.xp.float32)

layer_sizes = [X_train.shape[1], 16, 8, 1]

hidden_weight_initializer, hidden_bias_initializer = LayerInitializer.build_for_layer(
    layer_index=0,
    number_of_linear_layers=3,
    hidden_activation="relu",
    output_activation="sigmoid",
    hidden_bias_value=0.0,
)

output_weight_initializer, output_bias_initializer = LayerInitializer.build_for_layer(
    layer_index=2,
    number_of_linear_layers=3,
    hidden_activation="relu",
    output_activation="sigmoid",
    output_positive_prior=0.5,
)

model = ModelFactory.build_mlp(
    layer_sizes=layer_sizes,
    backend=backend,
    hidden_weight_initializer=hidden_weight_initializer,
    output_weight_initializer=output_weight_initializer,
    hidden_bias_initializer=hidden_bias_initializer,
    output_bias_initializer=output_bias_initializer,
    hidden_activation="relu",
    output_activation="sigmoid",
    dropout_probability=0.0,
)

print("Total parameters:", ModelFactory.count_parameters(model))

train_loader = BatchLoader(X_train, Y_train, batch_size=32, backend=backend, shuffle=True)
valid_loader = BatchLoader(X_valid, Y_valid, batch_size=32, backend=backend, shuffle=False)

loss_function = BinaryCrossEntropyLoss(backend)
optimizer = StochasticGradientDescent(model.parameters(), learning_rate=0.01)

config = TrainDefaults()
config.epochs = 10
config.threshold = 0.5

evaluator = Evaluator(model, backend)
finalizer = Finalizer(
    evaluator=evaluator,
    prediction_logger=PredictionLogger(),
    confusion_matrix_builder=ConfusionMatrix,
    error_analyzer=ErrorAnalysis,
)

trainer = Train(
    model=model,
    loss_function=loss_function,
    optimizer=optimizer,
    config=config,
    early_stopper=EarlyStopping(patience=5),
    best_model_tracker=ModelFactory(),
    finalizer=finalizer,
)

results = trainer.fit(
    train_loader=train_loader,
    validation_loader=valid_loader,
    epochs=config.epochs,
    restore_best=True,
    finalize=True,
    X_train_full=X_train,
    Y_train_full=Y_train,
    X_valid_full=X_valid,
    Y_valid_full=Y_valid,
    X_test_full=X_test,
    Y_test_full=Y_test,
    log_predictions=False,
    run_error_analysis=True,
)

print(results.train_losses)
print(results.validation_losses)
print(trainer.final_report)