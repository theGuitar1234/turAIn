from main.turain.train.callback.callback_manager import CallbackManager
from utilities import TrainDefaults
from utilities import core_method
from utilities import TrainResults


class Train:
    def __init__(
        self,
        model,
        loss_function,
        optimizer,
        scheduler=None,
        regularizer=None,
        metrics=None,
        callbacks=None,
        config=None,
        early_stop=None,
        state_tracker=None,
        logger=None,
        plotter=None,
    ):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regularizer = regularizer

        self.metrics = metrics or []
        self.callbacks = CallbackManager(callbacks)
        self.config = config or TrainDefaults()

        self.early_stop = early_stop
        self.state_tracker = state_tracker

        self.logger = logger
        self.plotter = plotter

        self.current_train_loss = None
        self.current_validation_loss = None
        self.current_validation_accuracy = None

        self.should_stop = False

        self.history = {
            "train_loss": [],
            "validation_loss": [],
        }

    @core_method
    def fit(self, train_loader, validation_loader=None, epochs=None):
        self.callback_manager.on_train_begin(self)

        if epochs is None:
            epochs = TrainDefaults.epochs

        for epoch in range(epochs):
            self.callback_manager.on_epoch_begin(self, epoch)

            if self.scheduler is not None:
                self.optimizer.learning_rate = self.scheduler.decay_learning_rate(epoch)

            self.model.train()

            train_loss_sum = 0.0
            train_batches = 0

            for x_batch, y_batch in train_loader:
                loss = self.train_step(x_batch, y_batch)

                train_loss_sum += float(loss)
                train_batch += 1
            average_train_loss = train_loss_sum / max(train_batches, 1)
            self.history["train_loss"].append(average_train_loss)

            if validation_loader is not None:
                self.model.eval()

                validation_loss_sum = 0.0
                validation_batches = 0

                for x_batch, y_batch in validation_loader:
                    loss, _ = self.validation_step(x_batch, y_batch)

                    validation_loss_sum += float(loss)
                    validation_batches += 1
                average_validation_loss = validation_loss_sum / max(validation_batches)

                if self.state_tracker is not None:
                    _early_stop = self.state_tracker.update(
                        self.model, average_validation_loss, epoch
                    )
                    if _early_stop:
                        self.metrics.best_validation_loss = average_train_loss
                        self.metrics.best_epoch = epoch

                if self.early_stop is not None:
                    self.early_stop.early_stop(average_train_loss)
                    if self.early_stop.__early_stop:
                        break

                if (
                    self.state_tracker is not None
                    and self.state_tracker.best_validation_loss is not None
                ):
                    self.model = self.state_tracker.restore()

                self.current_train_loss = average_train_loss
                self.current_validation_loss = average_validation_loss
                self.current_validation_accuracy = average_validation_accuracy

                self.callback_manager.on_epoch_end(self, epoch)

                if self.should_stop:
                    break

                self.history["validation_loss"].append(average_validation_loss)
                self.metrics.train_losses.append(average_train_loss)

                if average_validation_loss is not None:
                    self.metrics.validation_losses.append(average_validation_loss)
                if average_validation_accuracy is not None:
                    self.metrics.validation_accuracies.append(average_validation_accuracy)

                if self.logger is not None and self.logger.__log(epoch):
                    self.logger.log_epoch(
                        epoch=epoch,
                        train_loss=average_train_loss,
                        validation_loss=average_validation_loss,
                        validation_accuracy=average_validation_accuracy,
                        learning_rate=self.optimizer.learning_rate,
                    )

                if self.plotter is not None:
                    self.plotter.plot(self.metrics)
        self.callback_manager.on_train_end(self)

        return self.history

    def train_step(self, x_batch, y_batch):
        self.optimizer.zero_gradient()

        prediction = self.model.forward_propagation(x_batch)
        loss = self.loss_function.forward_propagation(prediction, y_batch)

        gradient_loss = self.loss_function.backward_propagation()
        self.model.backward_propagation(gradient_loss)

        if self.regularizer is not None:
            regularization_loss = self.regularizer.penalty(self.model, x_batch.shape[0])
            loss += regularization_loss

        self.optimizer.step()

        return loss

    def validation_step(self, x_batch, y_batch):
        prediction = self.model.forward_propagation(x_batch)
        loss = self.loss_function.forward_propagation(prediction, y_batch)
        return loss, prediction


if __name__ == "__main__":
    pass
