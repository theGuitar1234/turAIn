from dataclasses import dataclass


@dataclass
class TrainDefaults:
    learning_rate: float = 1e-2
    epochs: int = 5000
    reg: float = 1e-4
    epsilon: float = 1e-8
    step: int = 100
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
    momentum_coefficient: float = 0.9
    rms_coefficient: float = 0.999
    prediction_tolerance: int = 100
    prediction_threshold: int = 1000
    default_format_version: str = "1.0.0"


@dataclass
class TrainResults:
    losses: list = None
    val_losses: list = None
    best_val_loss: float = 0.0
    val_accuracies: list = None
    best_epoch: int = 0
    final_loss: float = 0.0
    accuracy: float = 0.0
    final_learning_rate: float = 0.0
    figure_title: str = "Training Results"
