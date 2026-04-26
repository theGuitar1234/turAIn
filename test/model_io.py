from turain.backend.cpu import CPU
from turain.models.sequential import Sequential
from turain.neural_network.layers.linear import Linear
from turain.neural_network.activations.relu import ReLU
from turain.data.model_io import ModelIO


def simple_weight_init(input_features, output_features, xp):
    return xp.random.standard_normal((output_features, input_features)) * 0.01


def simple_bias_init(output_features, xp):
    return xp.zeros((1, output_features))


backend = CPU()

model = Sequential(
    Linear(4, 8, backend, simple_weight_init, simple_bias_init),
    ReLU(backend),
    Linear(8, 2, backend, simple_weight_init, simple_bias_init),
)

metadata = ModelIO.build_metadata(model)
ModelIO.save_model(model, "models/test_model", metadata=metadata)

reloaded_model = Sequential(
    Linear(4, 8, backend, simple_weight_init, simple_bias_init),
    ReLU(backend),
    Linear(8, 2, backend, simple_weight_init, simple_bias_init),
)

reloaded_model, loaded_metadata = ModelIO.load_model(reloaded_model, "models/test_model.pkl")

print(loaded_metadata)
for p1, p2 in zip(model.parameters(), reloaded_model.parameters()):
    print((p1.data == p2.data).all())