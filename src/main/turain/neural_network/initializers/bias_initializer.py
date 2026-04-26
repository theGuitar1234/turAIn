from .distributions.bias.bias_constant import BiasConstant
from .distributions.bias.bias_standart_normal import BiasStandartNormal
from .distributions.bias.bias_uniform import BiasUniform
from .distributions.bias.bias_zeros import BiasZeros
from .initializer import Initializer

from ...utilities import TrainDefaults
from ...utilities import BiasInititializationStrategy
from ...utilities import OutputActivationType
from ...utilities import HiddenActivationType

from ...utilities import check_positive_integer, check_arguments

from ...lib import override_from_parent


class BiasInitialzer(Initializer):
    def __init__(
        self,
        backend,
        random_bias_initializing_strategy,
        is_output_layer,
        output_activation_type,
        hidden_activation_type,
    ):
        self.is_output_layer
        self.backend = backend
        self.random_bias_initializing_strategy = random_bias_initializing_strategy
        self.output_activation_type = output_activation_type
        self.is_output_layer = is_output_layer
        self.hidden_activation_type = hidden_activation_type

    @override_from_parent
    def initialize(self, fan_in, fan_out, config=None):
        b = None
        
        if config is None:
            config = TrainDefaults()
        output_positive_prior = config.output_positive_prior
        hidden_bias_value = config.hidden_bias_value
        bias_value = config.bias_value
        standart_deviation = config.standart_deviation

        xp = self.backend.xp
        rng = xp.random.default_rng()

        check_positive_integer(fan_out)

        if self.is_output_layer and output_positive_prior is not None:
            if self.output_activation_type != OutputActivationType.SIGMOID:
                raise ValueError(
                    "output_positive_prior is only supported for SIGMOID output layers"
                )

            check_arguments(
                int,
                (
                    fan_out,
                    {
                        "predicate": lambda fan_out: fan_out != 1,
                        "error_message": f"output_positive_prior requires {fan_out.__name__} == 1",
                    },
                ),
            )

            b0 = self.logit(output_positive_prior, config)
            b = xp.full((fan_out, 1), b0, dtype=float)

        if (
            not self.is_output_layer
            and self.hidden_activation_type == HiddenActivationType.RELU
            and hidden_bias_value != 0.0
        ):
            b = xp.full((fan_out, 1), hidden_bias_value, dtype=float)

        match self.random_bias_initializing_strategy:
            case BiasInititializationStrategy.ZERO:
                b = BiasZeros.__call__(xp, fan_out)

            case BiasInititializationStrategy.CONSTANT:
                b = BiasConstant.__call__(xp, fan_out, bias_value)

            case BiasInititializationStrategy.NORMAL:
                b = BiasStandartNormal.__call__(rng, fan_out, standart_deviation)

            case BiasInititializationStrategy.UNIFORM:
                b = BiasUniform.__call__(bias_value, fan_out, rng)

            case _:
                raise ValueError("Unknown Bias Init Strategy")
        
        return b

    def logit(self, p, config=None):
        xp = self.backend.xp
        if config is None:
            config = TrainDefaults()
        epsilon = config.epsilon
        p = float(xp.clip(p, epsilon, 1.0 - epsilon))
        return xp.log(p / (1.0 - p))
