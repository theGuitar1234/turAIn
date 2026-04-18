from utilities import WeightInitializationStrategy
from utilities import core_method


class Initializer:
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
                W = self.xavier_normal()
            case WeightInitializationStrategy.XAVIER_UNIFORM:
                W = self.xavier_uniform()
            case WeightInitializationStrategy.HE_NORMAL:
                W = self.he_normal()
            case WeightInitializationStrategy.HE_UNIFORM:
                W = self.he_uniform()
            case WeightInitializationStrategy.LECUN_NORMAL:
                W = self.lecun_normal()
            case WeightInitializationStrategy.LECUN_UNIFORM:
                W = self.lecun_uniform()
            case WeightInitializationStrategy.ZERO:
                W = self.zero()
            case _:
                raise ValueError(
                    f"Unknown Weight Init Strategy, supported values are : {list(WeightInitializationStrategy)}"
                )

    def xavier_normal(self, fan_in, fan_out, rng):
        xp = self.backend

        std = xp.sqrt(2.0 / (fan_in + fan_out))
        return rng.standard_normal((fan_out, fan_in)) * std

    def xavier_uniform(self, fan_in, fan_out, rng):
        xp = self.backend

        limit = xp.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_out, fan_in))
    
    def he_normal(self, fan_in, fan_out, alpha, rng):
        xp = self.backend
        
        std = xp.sqrt(2.0 / ((1.0 + alpha**2) * fan_in))
        return rng.standard_normal((fan_out, fan_in)) * std

    def he_uniform(self, alpha, fan_in, fan_out, rng, xp):
        limit = xp.sqrt(6.0 / ((1.0 + alpha**2) * fan_in))
        return rng.uniform(-limit, limit, size=(fan_out, fan_in))

    def lecun_normal(self, fan_in, fan_out, rng):
        xp = self.backend
        
        std = xp.sqrt(1.0 / fan_in)
        return rng.standard_normal((fan_out, fan_in)) * std

    def lecun_uniform(self, fan_in, fan_out, rng):
        xp = self.backend
        
        limit = xp.sqrt(3.0 / fan_in)
        return rng.uniform(-limit, limit, size=(fan_out, fan_in))

    def zero(self):
        xp = self.backend
        
        return xp.zeros((fan_out, fan_in))