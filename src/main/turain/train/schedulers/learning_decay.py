from utilities import core_method
from utilities import TrainDefaults


class LearningDecay:
    def __init__(self, initial_learning_rate, cfg):
        if cfg is None:
            cfg = TrainDefaults()
        self.decay_factor = cfg.decay_factor
        self.initial_learning_rate = initial_learning_rate

    @core_method
    def decay_learning_rate(self, epoch):
        raise NotImplementedError
