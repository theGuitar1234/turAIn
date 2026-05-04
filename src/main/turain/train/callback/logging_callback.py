from .callback import Callback


class LoggingCallback(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_epoch_end(self, trainer, epoch):
        self.logger.log_epoch(
            epoch=epoch,
            train_loss=trainer.current_train_loss,
            validation_loss=trainer.current_validation_loss,
            validation_accuracy=trainer.current_validation_accuracy,
            learning_rate=trainer.optimizer.learning_rate,
        )
