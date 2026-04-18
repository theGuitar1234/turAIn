from enum import Enum, auto


class Device(Enum):
    CPU = auto()
    CUDA = auto()


class Datasets(Enum):
    NPZ = auto()
    JSON = auto()
    PICKLE = auto()


class Optimizers(Enum):
    GDC = auto()
    MOMENTUM = auto()
    RMS_PROP = auto()
    ADAM = auto()


class LearningDecayType(Enum):
    STEP_DECAY = auto()
    INVERSE_DECAY = auto()
    EXPONENTIAL_DECAY = auto()


class OutputActivationType(Enum):
    SIGMOID = auto()
    SOFTMAX = auto()


class HiddenActivationType(Enum):
    SIGMOID = auto()
    RELU = auto()
    LEAKY_RELU = auto()
    TANH = auto()


class DataAugmentation(Enum):
    JITTER_NOISE = auto()
    SAME_CLASS_INTERPOLATION = auto()
    MEASUREMENT_NOISE = auto()


class LossType(Enum):
    MSE = auto()
    MULTI_CLASS_CROSS_ENTROPY = auto()
    BINARY_CROSS_ENTROPY = auto()


class BiasInititializationStrategy(Enum):
    ZERO = auto()
    CONSTANT = auto()
    NORMAL = auto()
    UNIFORM = auto()


class WeightInitializationStrategy(Enum):
    XAVIER_NORMAL = auto()
    XAVIER_UNIFORM = auto()
    HE_NORMAL = auto()
    HE_UNIFORM = auto()
    LECUN_NORMAL = auto()
    LECUN_UNIFORM = auto()
    ZERO = auto()


class StartWidthHeuristics(Enum):
    INPUT_WIDTH = auto()
    CAPPED_INPUT_WIDTH = auto()
    OUTPUT_AWARE = auto()


class LayerStrategies(Enum):
    CONSTANT_WIDTH = auto()
    LINEAR_TAPER_FUNNEL = auto()
    LINEAR_TAPER_FUNNEL_OUTPUT = auto()
    GEOMETRIC_TAPER = auto()
    EXPANSION_COMPRESSION = auto()
    BOTTLENECK_HOURGLASS = auto()
    POWER_OF_TWO = auto()
    REVERSE_POWER_OF_TWO = auto()
    PARAMETER_BUDGET = auto()
