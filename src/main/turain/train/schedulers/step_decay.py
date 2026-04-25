from learning_decay import LearningDecay
from lib import override_from_parent
from utilities import TrainDefaults


class StepDecay(LearningDecay):
    def __init__(self, initial_learning_rate, config=None):
        if config is None:
            config = TrainDefaults()
        super().__init__(initial_learning_rate, config)
        self.step_size = config.step_size

    @override_from_parent
    def decay_learning_rate(self, epoch):
        return self.initial_learning_rate * (self.decay_factor ** (epoch // self.step_size))



    
