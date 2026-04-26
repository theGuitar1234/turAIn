from main.turain.runtime.backend_selector import BackendSelector
from main.turain.runtime.device_types import DeviceTypes

from main.turain.data.dataset_io import DatasetIO
from main.turain.data.dataset_preparer import DatasetPreparer
from main.turain.data.batch_loader import BatchLoader

from main.turain.models.model_factory import ModelFactory
from main.turain.models.model_inspector import ModelInspector

from main.turain.neural_network.initializers.layer_initializer import LayerInitializer
from main.turain.neural_network.losses.binary_cross_entropy import BinaryCrossEntropyLoss

from main.turain.optimizers.stochastic_gradient_descent import StochasticGradientDescent

from main.turain.train.train import Train
from main.turain.train.train_config import TrainConfig
from main.turain.train.evaluator import Evaluator
from main.turain.train.finalizer import Finalizer
from main.turain.train.early_stopping import EarlyStopping
from main.turain.train.state_tracker import ModelStateTracker

from main.turain.metrics.confusion_matrix import ConfusionMatrix
from main.turain.metrics.error_analysis import ErrorAnalysis
from main.turain.metrics.prediction_logger import PredictionLogger


backend = BackendSelector.select(DeviceTypes.CPU)

dataset = DatasetIO.load_from_npz("data/npz/dataset.npz")
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

print("Total parameters:", ModelInspector.count_parameters(model))

train_loader = BatchLoader(X_train, Y_train, batch_size=32, backend=backend, shuffle=True)
valid_loader = BatchLoader(X_valid, Y_valid, batch_size=32, backend=backend, shuffle=False)

loss_function = BinaryCrossEntropyLoss(backend)
optimizer = StochasticGradientDescent(model.parameters(), learning_rate=0.01)

config = TrainConfig()
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
    best_model_tracker=ModelStateTracker(),
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