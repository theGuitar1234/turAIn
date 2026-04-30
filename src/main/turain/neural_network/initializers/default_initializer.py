from .distributions.weight.he_normal import HeNormal
from .distributions.weight.xavier_normal import XavierNormal
from ...utilities.enum import HiddenActivationType, OutputActivationType


class DefaultInitializer:
    @staticmethod
    def initialize_default_hidden_weight(hidden_activation_type):
        if hidden_activation_type in (HiddenActivationType.RELU, HiddenActivationType.LEAKY_RELU):
            return HeNormal()
        if hidden_activation_type in (HiddenActivationType.SIGMOID, HiddenActivationType.TANH):
            return XavierNormal()
        raise ValueError(f"Unknown hidden activation {hidden_activation_type}, supported values are {list(HiddenActivationType)}")

    @staticmethod
    def initialize_default_output_weight(output_activation_type):
        if output_activation_type in list(OutputActivationType):
            return XavierNormal()
        raise ValueError(f"Unknown output activation {output_activation_type}, supported values are {list(OutputActivationType)}")
