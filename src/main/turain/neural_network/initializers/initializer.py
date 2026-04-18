from utilities import WeightInitializationStrategy
from utilities import core_method

class Initializer:
    def __init__(
        self,
        input_features,
        output_features,
        random_hidden_weight_initializing_strategy,
        random_output_weight_initializing_strategy,
        random_bias_initializing_strategy,
        backend,
    ):
        self.input_features = (input_features,)
        self.output_features = (output_features,)
        self.random_hidden_weight_initializing_strategy = (
            random_hidden_weight_initializing_strategy,
        )
        self.random_output_weight_initializing_strategy = (
            random_output_weight_initializing_strategy,
        )
        self.random_bias_initializing_strategy = (random_bias_initializing_strategy,)
        self.backend = backend

    @core_method
    def initialize(self, xp, fan_in, fan_out, alpha):
        if rng is None:
            rng = xp.random.default_rng()

        if fan_in < 1 or fan_out < 1:
            raise ValueError("fan_in and fan_out must be positive integers")

        match self.random_hidden_weight_initializing_strategy:
            case WeightInitializationStrategy.XAVIER_NORMAL:
                std = xp.sqrt(2.0 / (fan_in + fan_out))
                return rng.standard_normal((fan_out, fan_in)) * std

            case WeightInitializationStrategy.XAVIER_UNIFORM:
                limit = xp.sqrt(6.0 / (fan_in + fan_out))
                return rng.uniform(-limit, limit, size=(fan_out, fan_in))

            case WeightInitializationStrategy.HE_NORMAL:
                std = xp.sqrt(2.0 / ((1.0 + alpha**2) * fan_in))
                return rng.standard_normal((fan_out, fan_in)) * std

            case WeightInitializationStrategy.HE_UNIFORM:
                limit = xp.sqrt(6.0 / ((1.0 + alpha**2) * fan_in))
                return rng.uniform(-limit, limit, size=(fan_out, fan_in))

            case WeightInitializationStrategy.LECUN_NORMAL:
                std = xp.sqrt(1.0 / fan_in)
                return rng.standard_normal((fan_out, fan_in)) * std

            case WeightInitializationStrategy.LECUN_UNIFORM:
                limit = xp.sqrt(3.0 / fan_in)
                return rng.uniform(-limit, limit, size=(fan_out, fan_in))

            case WeightInitializationStrategy.ZERO:
                return xp.zeros((fan_out, fan_in))

            case _:
                raise ValueError(
                    f"Unknown Weight Init Strategy, supported values are : {list(WeightInitializationStrategy)}"
                )
