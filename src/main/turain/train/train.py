from callback import CallbackManager

from utilities import TrainDefaults
from utilities import TrainResults

from utilities import core_method, helper_method
from lib import cpu_engine


class Train:
    def __init__(
        self,
        model,
        loss_function,
        optimizer,
        scheduler=None,
        regularizer=None,
        metrics=None,
        callback_manager=None,
        config=None,
        early_stop=None,
        state_tracker=None,
        logger=None,
        plotter=None,
        finalizer=None,
    ):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.scheduler = scheduler
        self.regularizer = regularizer

        self.metrics = metrics or []
        self.callback_manager = CallbackManager(callback_manager)
        self.config = config or TrainDefaults()
        self.results = TrainResults()

        self.early_stop = early_stop
        self.state_tracker = state_tracker

        self.logger = logger
        self.plotter = plotter
        self.finalizer = finalizer

        self.current_train_loss = None
        self.current_validation_loss = None
        self.current_validation_accuracy = None

    @core_method
    def fit(
        self,
        train_loader,
        validation_loader=None,
        epochs=None,
        track_state=False,
    ):
        if epochs is None:
            epochs = TrainDefaults.epochs

        if self.callback_manager is not None:
            self.callback_manager.on_train_begin(self)

        for epoch in range(epochs):
            if self.callback_manager is not None:
                self.callback_manager.on_epoch_begin(self, epoch)

            if self.scheduler is not None:
                self.optimizer.learning_rate = self.scheduler.decay_learning_rate(epoch)

            self.model.train()

            train_loss_sum = 0.0
            train_batches = 0

            for x_batch, y_batch in train_loader:
                batch_loss = self.train_step(x_batch, y_batch)
                batch_size = x_batch.shape[0]

                train_loss_sum += self.to_python_scalar(batch_loss) * batch_size
                train_samples += batch_size
            average_train_loss = train_loss_sum / max(train_batches, 1)

            self.current_train_loss = average_train_loss
            self.current_validation_loss = None
            self.current_validation_accuracy = None

            self.results.train_losses.append(average_train_loss)

            if validation_loader is not None:
                self.model.eval()

                total_validation_loss = 0.0
                total_validation_accuracy = 0.0
                validation_samples = 0

                for x_batch, y_batch in validation_loader:
                    validation_batch_loss, prediction = self.validation_step(x_batch, y_batch)

                    batch_validation_accuracy = self._batch_accuracy(
                        prediction, y_batch, threshold=self.config.threshold
                    )

                    total_validation_loss += self.to_python_scalar(batch_loss)
                    total_validation_accuracy += batch_validation_accuracy * batch_size
                    validation_samples += batch_size
                average_validation_loss = total_validation_loss / max(validation_samples, 1)
                average_validation_accuracy = total_validation_accuracy / max(validation_samples, 1)

                self.current_validation_loss = average_validation_loss
                self.current_validation_accuracy = average_validation_accuracy

                self.results.validation_losses.append(average_validation_loss)
                self.results.validation_accuracies.append(average_validation_accuracy)

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

                if (
                    self.state_tracker is not None
                    and self.state_tracker.best_validation_loss is not None
                ):
                    self.model = self.state_tracker.restore()

                self.current_train_loss = average_train_loss
                self.current_validation_loss = average_validation_loss
                self.current_validation_accuracy = average_validation_accuracy

                if self.callback_manager is not None:
                    self.callback_manager.on_epoch_end(self, epoch)

                if self.state_tracker is not None:
                    _early_stop = self.state_tracker.update(
                        self.model, average_validation_loss, epoch
                    )
                    if _early_stop:
                        self.results.best_validation_loss = average_train_loss
                        self.results.best_epoch = epoch

                if self.early_stop is not None:
                    self.early_stop.early_stop(average_train_loss)
                    if self.early_stop.__early_stop:
                        break

            if track_state and self.state_tracker is not None:
                best_model = self.state_tracker.restore()
                if best_model is not None:
                    self.model = best_model

            self.results.final_learning_rate = self.optimizer.learning_rate

            if self.plotter is not None and hasattr(self.plotter, "plot"):
                self.plotter.plot(self.results)

            if self.callback_manager is not None:
                self.callback_manager.on_train_end(self)

            if self.finalizer is not None:
                final_report = self.finalizer.finalize(
                    model=self.model,
                    loss_function=self.loss_function,
                    threshold=self.config.threshold,
                    log_predictions=False,
                    run_error_analysis=False,
                )

            self.results.final_loss = (
                final_report.train_loss if final_report is not None else self.results.final_loss
            )
            self.results.final_accuracy = (
                final_report.train_accuracy
                if final_report is not None
                else self.results.final_accuracy
            )

        return self.results

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

    @helper_method
    def _batch_accuracy(self, prediction, true_label, threshold=None):
        backend = self._get_backend()
        xp = backend.xp

        if threshold is None:
            threshold = self.config.threshold

        if prediction.ndim == 2 and prediction.shape[1] == 1:
            predicted = (prediction >= threshold).astype(prediction.dtype)

            if true_label.ndim == 2 and true_label.shape[1] == 1:
                true_values = true_label
            else:
                true_values = true_label.reshape(-1, 1)

            accuracy = xp.mean(predicted == true_values) * 100.0
            return self._to_python_scalar(accuracy)

        predicted_indices = xp.argmax(prediction, axis=1)

        if true_label.ndim == 2 and true_label.shape[1] > 1:
            true_indices = xp.argmax(true_label, axis=1)
        else:
            true_indices = true_label.reshape(-1)

        accuracy = xp.mean(predicted_indices == true_indices) * 100.0
        return self._to_python_scalar(accuracy)

    @helper_method
    def _get_backend(self):
        for layer in self.model.layers:
            if hasattr(layer, "backend"):
                return layer.backend
        raise RuntimeError("Could not determine backend from model layers.")

    @helper_method
    def _to_python_scalar(self, value):
        backend = self._get_backend()
        cpu_value = backend.to_cpu(value)
        return float(cpu_engine.asarray(cpu_value))
