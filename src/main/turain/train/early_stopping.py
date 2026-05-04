from ..utilities import core_method


class EarlyStopping:
    def __init__(self, patience=None, minimum_validation_loss_delta=0.0):
        self.patience = patience
        self.minimum_validation_loss_delta = minimum_validation_loss_delta
        self.best_validation_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    @core_method
    def early_stop(self, validation_loss):
        _early_stop = validation_loss < (self.best_validation_loss - self.minimum_validation_loss_delta)
        if _early_stop: 
            self.best_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop