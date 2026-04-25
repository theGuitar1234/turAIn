from .initializer import Initializer

from utilities import WeightInitializationStrategy
from utilities import core_method, check_positive_integer
from lib import override_from_parent

from distributions.weight.he_normal import HeNormal
from distributions.weight.he_uniform import HeUniform
from distributions.weight.lecun_normal import LeCunNormal
from distributions.weight.lecun_uniform import LeCunUniform
from distributions.weight.xavier_normal import XavierNormal
from distributions.weight.xavier_uniform import XavierUniform
from distributions.weight.zero import Zero


class WeightInitializer(Initializer):
    def __init__(
        self,
        layer,
        backend,
        number_of_neurons,
        input_features,
        output_width,
        random_hidden_weight_initializing_strategy,
        random_output_weight_initializing_strategy,
        random_bias_initializing_strategy
    ):
        super().__init__(backend, output_width)
        
        self.input_features = input_features
        self.random_hidden_weight_initializing_strategy = (
            random_hidden_weight_initializing_strategy,
        )
        self.random_output_weight_initializing_strategy = (
            random_output_weight_initializing_strategy,
        )
        self.random_bias_initializing_strategy = (random_bias_initializing_strategy,)

        self.fan_in = input_features if layer == 0 else output_width
        self.fan_out = number_of_neurons

    @override_from_parent
    def initialize(self, xp, fan_in, fan_out, alpha):
        W = None
        
        rng = xp.random.default_rng()

        check_positive_integer(fan_in, fan_out)

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
                raise ValueError(f"Unknown Weight Init Strategy, supported values are : {list(WeightInitializationStrategy)}")
        return W


if __name__ == "__main__":
    pass
