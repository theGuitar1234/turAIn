from turain.backend.cpu import CPU
from turain.neural_network.initializers.bias_initializer import BiasInitialzer
from turain.neural_network.initializers.default_initializer import DefaultInitializer
from turain.utilities.enum import BiasInititializationStrategy, HiddenActivationType, OutputActivationType

backend = CPU()
xp = backend.xp

initializer = BiasInitialzer(
    backend,
    random_bias_initializing_strategy=BiasInititializationStrategy.CONSTANT,
    is_output_layer=False,
    output_activation_type=OutputActivationType.SIGMOID,
    hidden_activation_type=HiddenActivationType.RELU,
)

bias = initializer.initialize(4, 8)
print(bias.shape)
print(bias)

print(type(DefaultInitializer.initialize_default_hidden_weight(HiddenActivationType.RELU)).__name__)
print(type(DefaultInitializer.initialize_default_hidden_weight(HiddenActivationType.TANH)).__name__)
print(type(DefaultInitializer.initialize_default_output_weight(OutputActivationType.SIGMOID)).__name__)
