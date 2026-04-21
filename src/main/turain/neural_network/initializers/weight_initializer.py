from utilities import WeightInitializationStrategy
from utilities import core_method

from main.turain.neural_network.initializers.distributions.he_normal import HeNormal
from main.turain.neural_network.initializers.distributions.he_uniform import HeUniform
from main.turain.neural_network.initializers.distributions.lecun_normal import LeCunNormal
from main.turain.neural_network.initializers.distributions.lecun_uniform import LeCunUniform
from main.turain.neural_network.initializers.distributions.xavier_normal import XavierNormal
from main.turain.neural_network.initializers.distributions.xavier_uniform import XavierUniform
from main.turain.neural_network.initializers.distributions.zero import Zero


class WeightInitializer:
    def __init__(
        self,
        layer,
        number_of_neurons,
        input_features,
        output_features,
        random_hidden_weight_initializing_strategy,
        random_output_weight_initializing_strategy,
        random_bias_initializing_strategy,
        backend,
    ):
        self.input_features = input_features
        self.output_features = output_features
        self.random_hidden_weight_initializing_strategy = (
            random_hidden_weight_initializing_strategy,
        )
        self.random_output_weight_initializing_strategy = (
            random_output_weight_initializing_strategy,
        )
        self.random_bias_initializing_strategy = (random_bias_initializing_strategy,)
        self.backend = backend

        self.fan_in = input_features if layer == 0 else output_features
        self.fan_out = number_of_neurons

    @core_method
    def initialize(self, xp, fan_in, fan_out, alpha):
        if rng is None:
            rng = xp.random.default_rng()

        if fan_in < 1 or fan_out < 1:
            raise ValueError("fan_in and fan_out must be positive integers")

        match self.random_hidden_weight_initializing_strategy:
            case WeightInitializationStrategy.XAVIER_NORMAL:
                W = XavierNormal.__call__(fan_in, fan_out, xp)
            case WeightInitializationStrategy.XAVIER_UNIFORM:
                W = XavierUniform.__call__(fan_in, fan_out, rng, xp)
            case WeightInitializationStrategy.HE_NORMAL:
                W = HeNormal.__call__(fan_in, fan_out, alpha, rng, xp)
            case WeightInitializationStrategy.HE_UNIFORM:
                W = HeUniform.__call__(alpha, fan_in, fan_out, rng, xp)
            case WeightInitializationStrategy.LECUN_NORMAL:
                W = LeCunNormal.__call__(fan_in, fan_out, rng, xp)
            case WeightInitializationStrategy.LECUN_UNIFORM:
                W = LeCunUniform.__call__(fan_in, fan_out, rng, xp)
            case WeightInitializationStrategy.ZERO:
                W = Zero.__call__(fan_in, fan_out, xp)
            case _:
                raise ValueError(
                    f"Unknown Weight Init Strategy, supported values are : {list(WeightInitializationStrategy)}"
                )
        return W


if __name__ == "__main__":
    pass
