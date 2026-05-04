from turain.utilities.enum import HiddenActivationType

from .initializer import Initializer

from ...utilities import WeightInitializationStrategy
from ...utilities import core_method, check_positive_integer
from ...lib import override_from_parent

from .distributions.weight.he_normal import HeNormal
from .distributions.weight.he_uniform import HeUniform
from .distributions.weight.lecun_normal import LeCunNormal
from .distributions.weight.lecun_uniform import LeCunUniform
from .distributions.weight.xavier_normal import XavierNormal
from .distributions.weight.xavier_uniform import XavierUniform
from .distributions.weight.zero import Zero


class WeightInitializer(Initializer):
    def __init__(
        self,
        layer,
        backend,
        is_output_layer,
        number_of_neurons,
        number_of_features,
        output_features,
        hidden_weight_initializing_strategy,
        output_weight_initializing_strategy,
        bias_initializing_strategy,
        output_activation_type,
        hidden_activation_type,
    ):
        super().__init__(backend, output_features)

        self.input_features = number_of_features
        self.hidden_weight_initializing_strategy = hidden_weight_initializing_strategy
        self.output_weight_initializing_strategy = output_weight_initializing_strategy
        self.bias_initializing_strategy = bias_initializing_strategy
        self.layer = layer  
        self.number_of_features = number_of_features
        self.number_of_neurons = number_of_neurons
        self.output_features = output_features
        
        alpha = 0.0
        if (
            not is_output_layer
            and hidden_activation_type == HiddenActivationType.LEAKY_RELU
        ):
            alpha = 0.01
        self.alpha = alpha

    @override_from_parent
    def initialize(self):
        xp = self.backend.xp

        W = None

        rng = xp.random.default_rng()
        
        fan_in = self.number_of_features
        fan_out = self.number_of_neurons

        check_positive_integer(fan_in, fan_out)

        match self.hidden_weight_initializing_strategy:
            case WeightInitializationStrategy.XAVIER_NORMAL:
                W = XavierNormal.__call__(fan_in, fan_out, xp)
            case WeightInitializationStrategy.XAVIER_UNIFORM:
                W = XavierUniform.__call__(fan_in, fan_out, rng, xp)
            case WeightInitializationStrategy.HE_NORMAL:
                W = HeNormal.__call__(fan_in, fan_out, self.alpha, rng, xp)
            case WeightInitializationStrategy.HE_UNIFORM:
                W = HeUniform.__call__(self.alpha, fan_in, fan_out, rng, xp)
            case WeightInitializationStrategy.LECUN_NORMAL:
                W = LeCunNormal.__call__(fan_in, fan_out, rng, xp)
            case WeightInitializationStrategy.LECUN_UNIFORM:
                W = LeCunUniform.__call__(fan_in, fan_out, rng, xp)
            case WeightInitializationStrategy.ZERO:
                W = Zero.__call__(fan_in, fan_out, xp)
            case _:
                raise ValueError(
                    f"Unknown Weight Init Strategy {self.hidden_weight_initializing_strategy}, supported values are : {list(WeightInitializationStrategy)}"
                )
        return W
