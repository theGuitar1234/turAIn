from .distributions.layer.bottleneck_hourglass import BottleNeckHourGlass
from .distributions.layer.constant_width import ConstantWidth
from .distributions.layer.expansion_compression import ExpansionCompression
from .distributions.layer.geometric_taper import GeometricTaper
from .distributions.layer.linear_taper_funnel import LinearTaperFunnel
from .distributions.layer.linear_taper_funnel_output import LinearTaperFunnelOutput
from .distributions.layer.parameter_budget import ParameterBudget
from .distributions.layer.powers_of_two import PowersOfTwo
from .distributions.layer.reverse_power_of_two import ReversePowerOfTwo

from .distributions.layer.start_width import CappedInputWidth
from .distributions.layer.start_width import InputWidth
from .distributions.layer.start_width import OutputAware

from .initializer import Initializer

from ...lib import override_from_parent

from ...utilities import TrainDefaults
from ...utilities import LayerStrategies
from ...utilities import StartWidthHeuristics
from ...utilities import check_positive_integer
from ...utilities import helper_method


class LayerInitializer(Initializer):
    def __init__(
        self,
        backend,
        number_of_hidden_layers,
        output_width=None,
        start_width=None,
        hidden_width=None,
        layer_strategy=None,
        start_width_heuristic=None,
        start_width_heuristic_cap=None,
        output_aware_multiplier=None,
    ):
        super().__init__(backend, output_width)

        if layer_strategy is None:
            print(
                f"No Layer Strategy is chosen, falling back to the default {LayerStrategies.GEOMETRIC_TAPER.name}... "
            )
            layer_strategy = LayerStrategies.GEOMETRIC_TAPER
        if start_width_heuristic is None:
            print(
                f"No Start Width Heuristic is chosen, falling back to the default {StartWidthHeuristics.OUTPUT_AWARE.name}... "
            )
            start_width_heuristic = StartWidthHeuristics.OUTPUT_AWARE
        
        if layer_strategy is LayerStrategies.LINEAR_TAPER_FUNNEL:
            if hidden_width is None:
                print(f"For the strategy {LayerStrategies.LINEAR_TAPER_FUNNEL.name}, a hidden_width is required, falling back to a safe initialization...")
                hidden_width = max(output_width * 2, start_width // 2)

        check_positive_integer(number_of_hidden_layers, start_width, hidden_width)

        self.number_of_hidden_layers = number_of_hidden_layers
        self.layer_strategy = layer_strategy
        self.start_width = start_width
        self.hidden_width = hidden_width
        self.start_width_heuristic = start_width_heuristic
        self.start_width_heuristic_cap = start_width_heuristic_cap

    @override_from_parent
    def initialize(self, X, config=None):
        if self.number_of_hidden_layers == 0:
            print(f"No Hidden Layers are defined, initializing only the output layer...")
            return [self.output_width]

        if config is None:
            config = TrainDefaults()
        expansion_multiplier = config.expansion_multiplier
        parameter_budget = config.parameter_budget
        
        if self.layer_strategy is LayerStrategies.PARAMETER_BUDGET:
            check_positive_integer(parameter_budget)

        xp = self.backend.xp
        if not isinstance(X, xp.ndarray):
            raise TypeError("X must be a numpy.ndarray")
        if X.ndim != 2:
            raise ValueError("X must be a 2D feature-first matrix")

        input_width = X.shape[0]

        if self.start_width is None:
            self.set_start_width(self.start_width_heuristic, self.start_width_heuristic_cap, input_width, self.output_width)

        match self.layer_strategy:
            case LayerStrategies.CONSTANT_WIDTH:
                layers = ConstantWidth.__call__(self.start_width, self.number_of_hidden_layers)
            case LayerStrategies.LINEAR_TAPER_FUNNEL:
                layers = LinearTaperFunnel.__call__(
                    self.start_width, self.number_of_hidden_layers, self.hidden_width
                )
            case LayerStrategies.LINEAR_TAPER_FUNNEL_OUTPUT:
                layers = LinearTaperFunnelOutput.__call__(
                    self.start_width, self.number_of_hidden_layers, self.output_width
                )
            case LayerStrategies.GEOMETRIC_TAPER:
                layers = GeometricTaper.__call__(
                    self.start_width, self.output_width, self.number_of_hidden_layers
                )
            case LayerStrategies.EXPANSION_COMPRESSION:
                layers = ExpansionCompression.__call__(
                    self.start_width,
                    input_width,
                    expansion_multiplier,
                    self.number_of_hidden_layers,
                    self.output_width,
                )
            case LayerStrategies.BOTTLENECK_HOURGLASS:
                layers = BottleNeckHourGlass.__call__(
                    self.start_width, self.output_width, self.number_of_hidden_layers
                )
            case LayerStrategies.POWER_OF_TWO:
                layers = PowersOfTwo.__call_(
                    self.start_width, self.output_width, self.number_of_hidden_layers, xp
                )
            case LayerStrategies.REVERSE_POWER_OF_TWO:
                layers = ReversePowerOfTwo.__call(
                    self.start_width, self.output_width, self.number_of_hidden_layers
                )
            case LayerStrategies.PARAMETER_BUDGET:
                layers = ParameterBudget.__call__(
                    input_width, self.output_width, self.number_of_hidden_layers, parameter_budget
                )
            case _:
                raise ValueError(f"Unknown Layer Strategy {self.layer_strategy}, supported values are: {list(LayerStrategies)}")

        return layers + [self.output_width]

    @helper_method
    def set_start_width(self, start_width_heuristic, start_width_heuristic_cap, input_width, output_width):
        match start_width_heuristic:
            case StartWidthHeuristics.INPUT_WIDTH:
                start_width = InputWidth.__call__(input_width)
            case StartWidthHeuristics.CAPPED_INPUT_WIDTH:
                start_width = CappedInputWidth.__call__(start_width_heuristic_cap, input_width)
            case StartWidthHeuristics.OUTPUT_AWARE:
                start_width = OutputAware.__call__(output_width, input_width, start_width_heuristic_cap)
            case _:
                raise ValueError(f"Unknown Start Width Heuristic {self.start_width_heuristic}, supported values are : {list(StartWidthHeuristics)}")
        return start_width



    
