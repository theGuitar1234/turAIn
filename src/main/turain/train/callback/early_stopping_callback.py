from .callback import Callback


class EarlyStoppingCallback(Callback):
    def __init__(self, early_stopper):
        self.early_stopper = early_stopper

    def on_epoch_end(self, trainer, epoch):
        validation_loss = trainer.current_validation_loss
        if validation_loss is None:
            return

        self.early_stopper.update(validation_loss)

        if self.early_stopper.should_stop:
            trainer.should_stop = True
