from main.turain.data.dataset_preparer import DatasetPreparer
from main.turain.data.deserialization import DeSerialization
from main.turain.models.model_factory import ModelFactory
from main.turain.neural_network.initializers.layer_initializer import LayerInitializer
from main.turain.neural_network.losses.softmax import SoftmaxCrossEntropyLoss
from main.turain.runtime.backend_selector import BackendSelector
from main.turain.utilities.enum import Device

backend = BackendSelector.select(Device.CPU)
dataset = DeSerialization.load_from_npz("data/npz/dataset.npz")

number_of_classes = 10

prepared = DatasetPreparer.prepare(
    X_train=dataset["X_train"],
    Y_train=dataset["Y_train"],
    X_valid=dataset["X_valid"],
    Y_valid=dataset["Y_valid"],
    X_test=dataset["X_test"],
    Y_test=dataset["Y_test"],
    flatten_features=False,
    one_hot=True,
    number_of_classes=number_of_classes,
    cast_labels_to_int=True,
)

X_train = prepared["X_train"]
layer_sizes = [X_train.shape[1], 32, 16, number_of_classes]

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
    output_activation=None,
    dropout_probability=0.0,
)

loss_function = SoftmaxCrossEntropyLoss(backend)