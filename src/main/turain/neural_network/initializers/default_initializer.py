from .distributions.weight.he_normal import HeNormal
from .distributions.weight.xavier_normal import XavierNormal
from ...utilities.enum import HiddenActivationType, OutputActivationType


class DefaultInitializer:
    @staticmethod
    def initialize_default_hidden_weight(hidden_activation):
        if hidden_activation in (HiddenActivationType.RELU, HiddenActivationType.LEAKY_RELU):
            return HeNormal()
        if hidden_activation in (HiddenActivationType.SIGMOID, HiddenActivationType.TANH):
            return XavierNormal()
        raise ValueError(
            f"Unknown hidden activation, supported values are {list(HiddenActivationType)}"
        )

    @staticmethod
    def initialize_default_output_weight(output_activation):
        if output_activation in list(OutputActivationType):
            return XavierNormal()
        raise ValueError(
            f"Unknown output activation, supported values are {list(OutputActivationType)}"
        )
