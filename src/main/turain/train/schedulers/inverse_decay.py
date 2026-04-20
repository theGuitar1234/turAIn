from learning_decay import LearningDecay
from lib import override_from_parent


class InverseDecay(LearningDecay):
    def __init__(self, initial_learning_rate, cfg=None):
        super().__init__(initial_learning_rate, cfg)

    @override_from_parent
    def decay_learning_rate(self, epoch):
        return self.initial_learning_rate / (1.0 + self.decay_factor * epoch)
