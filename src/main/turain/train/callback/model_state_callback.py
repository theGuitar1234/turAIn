from main.turain.train.callback import Callback


class BestModelCallback(Callback):
    def __init__(self, best_model_tracker):
        self.best_model_tracker = best_model_tracker

    def on_epoch_end(self, trainer, epoch):
        validation_loss = trainer.current_validation_loss
        if validation_loss is None:
            return

        improved = self.best_model_tracker.update(trainer.model, validation_loss, epoch)

        if improved:
            trainer.results.best_validation_loss = validation_loss
            trainer.results.best_epoch = epoch

    def on_train_end(self, trainer):
        if self.best_model_tracker.best_state is not None:
            trainer.model = self.best_model_tracker.restore()
