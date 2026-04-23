from lib import dto, mutable_field

@dto
class TrainDefaults:
    learning_rate: float = 1e-2
    epochs: int = 5000
    epsilon: float = 1e-8
    step_size: int = 100
    threshold: float = 0.5
    drop_out_rate: float = 0.03
    l2_lambda: float = 0.03
    batch_size: int = 512
    patience: int = 100
    decay_factor: float = 0.5
    seed: int = 42
    start_width_heuristic_cap: int = 512
    output_aware_multiplier: int = 4
    expansion_multiplier: int = 2
    parameter_budget: int = 2
    momentum_coefficient: float = 0.9
    measurement_strength: float = 0.02
    rms_coefficient: float = 0.999
    negative_slope: int = 0.01
    prediction_tolerance: int = 100
    prediction_threshold: int = 1000
    default_format_version: str = "1.0.0"


@dto
class TrainResults:
    train_losses: list = mutable_field(default_factory=list)
    validation_losses: list = mutable_field(default_factory=list)
    validation_accuracies: list = mutable_field(default_factory=list)

    best_validation_loss: float = float("inf")
    best_epoch: int | None = None

    final_loss: float | None = None
    final_accuracy: float | None = None
    final_learning_rate: float | None = None

    title: str = "Training Results"