from utilities import core_method
import copy


class BestStateTracker:
    def __init__(self):
        self.best_validation_loss = float("inf")
        self.best_epoch = None
        self.best_state = None

    @core_method
    def update(self, model, validation_loss, epoch):
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model)
            return True
        return False

    def restore(self):
        return copy.deepcopy(self.best_validation_loss) if self.best_state is not None else None
