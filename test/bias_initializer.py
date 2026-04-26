from turain.backend.cpu import CPU
from turain.neural_network.initializers.bias_initializer import BiasInitializer
from turain.neural_network.initializers.default_initializer import DefaultInitializer

backend = CPU()
xp = backend.xp

initializer = BiasInitializer(
    strategy="zero",
    is_output_layer=False,
    hidden_activation="relu",
    hidden_bias_value=0.25,
)

bias = initializer(8, xp)
print(bias.shape)
print(bias)

print(type(DefaultInitializer.initialize_default_hidden_weight("relu")).__name__)
print(type(DefaultInitializer.initialize_default_hidden_weight("tanh")).__name__)
print(type(DefaultInitializer.initialize_default_output_weight("sigmoid")).__name__)
