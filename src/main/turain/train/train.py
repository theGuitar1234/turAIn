from ..utilities import TrainDefaults
from ..utilities import TrainResults

from ..utilities import core_method, helper_method
from ..lib import cpu_engine
from ..lib import plotting

class Train:
    def __init__(
        self,
        model,
        loss_function,
        optimizer,
        scheduler=None,
        l2_regularizer=None,
        l1_regularizer=None,
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
        self.l2_regularizer = l2_regularizer
        self.l1_regularizer = l1_regularizer

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
        X_train=None,
        Y_train=None,
        X_valid=None,
        Y_valid=None,
        X_test=None,
        Y_test=None,
        train_loader=None,
        validation_loader=None,
        track_state=False,
        finalize=False,
        log_predictions=False,
        run_error_analysis=False,
        plot=False,
        plot_real_time=False,
        epochs=None,
        config=None,
    ):
        self.check_flags(
            ("log_predictions", log_predictions, "Logger", self.logger),
            ("plot_real_time", plot_real_time, "Plotter", self.plotter),
            ("plot", plot, "Plotter", self.plotter),
            ("track_state", track_state, "StateTracker", self.state_tracker),
            ("finalize", finalize, "Finalizer", self.finalizer),
        )

        if config is None:
            config = TrainDefaults()
        if epochs is None:
            epochs = config.epochs
        if (plot or plot_real_time) and self.plotter is not None:
            self.plotter.settle_plot(self.results)
        if plot_real_time and self.plotter is not None:
            plotting.ion()
        final_report = None
        self.should_stop = False

        for epoch in range(epochs):
            if self.scheduler is not None:
                self.optimizer.learning_rate = self.scheduler.decay_learning_rate(epoch)
            self.model.train()

            total_train_loss = 0.0
            total_validation_loss = 0.0
            validation_batch_loss = 0.0
            total_validation_accuracy = 0.0
            batch_validation_accuracy = 0.0
            train_samples = 0
            validation_samples = 0

            for x_batch, y_batch in train_loader:
                batch_loss = self.train_step(x_batch, y_batch)
                batch_size = x_batch.shape[0]

                total_train_loss += self._to_python_scalar(batch_loss) * batch_size
                train_samples += batch_size
            average_train_loss = total_train_loss / max(train_samples, 1)

            self.current_train_loss = average_train_loss
            self.current_validation_loss = None
            self.current_validation_accuracy = None

            self.results.train_losses.append(average_train_loss)

            if validation_loader is not None:
                self.model.evaluate()

                batch_size = x_batch.shape[0]
                total_validation_loss += self._to_python_scalar(validation_batch_loss) * batch_size
                total_validation_accuracy += batch_validation_accuracy * batch_size
                validation_samples += batch_size

                for x_batch, y_batch in validation_loader:
                    validation_batch_loss, prediction = self.validation_step(x_batch, y_batch)

                    batch_validation_accuracy = self._batch_accuracy(
                        prediction, y_batch, threshold=config.threshold
                    )

                    total_validation_loss += self._to_python_scalar(batch_loss)
                    total_validation_accuracy += batch_validation_accuracy * batch_size
                    validation_samples += batch_size
                average_validation_loss = total_validation_loss / max(validation_samples, 1)
                average_validation_accuracy = total_validation_accuracy / max(validation_samples, 1)

                self.current_validation_loss = average_validation_loss
                self.current_validation_accuracy = average_validation_accuracy

                self.results.validation_losses.append(average_validation_loss)
                self.results.validation_accuracies.append(average_validation_accuracy)

                if log_predictions and self.logger is not None:
                    self.logger.log_epoch(
                        epoch=epoch,
                        train_loss=average_train_loss,
                        validation_loss=average_validation_loss,
                        validation_accuracy=average_validation_accuracy,
                        learning_rate=self.optimizer.learning_rate,
                    )
                if plot_real_time and self.plotter is not None:
                    self.plotter.plot_real_time(self.results)
                if self.state_tracker is not None:
                    improved = self.state_tracker.update(self.model, average_validation_loss, epoch)
                    if improved:
                        self.results.best_validation_loss = average_validation_loss
                        self.results.best_epoch = epoch
                self.current_train_loss = average_train_loss
                self.current_validation_loss = average_validation_loss
                self.current_validation_accuracy = average_validation_accuracy

                if self.state_tracker is not None:
                    _early_stop = self.state_tracker.update(
                        self.model, average_validation_loss, epoch
                    )
                    if _early_stop:
                        self.results.best_validation_loss = average_validation_loss
                        self.results.best_epoch = epoch
                if self.early_stop is not None:
                    if self.early_stop.early_stop(average_validation_loss):
                        self.should_stop = True
            if track_state and self.state_tracker is not None:
                best_model = self.state_tracker.restore()
                if best_model is not None:
                    self.model = best_model
            if self.should_stop:
                break
        self.results.final_learning_rate = self.optimizer.learning_rate

        if finalize and self.finalizer is not None:
            finalizer_report = self.finalizer.finalize(
                X_train=X_train,
                Y_train=Y_train,
                X_valid=X_valid,
                Y_valid=Y_valid,
                X_test=X_test,
                Y_test=Y_test,
                run_error_analysis=run_error_analysis,
                log_predictions=log_predictions,
                config=config,
            )
            self.results.finalizer_report = finalizer_report

        if self.results is not None:
            self.results.final_loss = (
                final_report.train_loss if final_report is not None else self.results.final_loss
            )
            self.results.final_accuracy = (
                final_report.train_accuracy
                if final_report is not None
                else self.results.final_accuracy
            )

        if plot and self.plotter is not None:
            self.plotter.plot_once(self.results)
        if track_state and self.state_tracker is not None:
            best_model = self.state_tracker.restore()
            if best_model is not None:
                self.model = best_model
        return self.results

    def train_step(self, x_batch, y_batch):
        self.optimizer.zero_gradient()

        prediction = self.model.forward_propagation(x_batch)
        loss = self.loss_function.forward_propagation(prediction, y_batch)

        gradient_loss = self.loss_function.backward_propagation()
        self.model.backward_propagation(gradient_loss)

        if self.l2_regularizer is not None:
            regularization_loss = self.l2_regularizer.penalty(self.model, x_batch.shape[0])
            loss += regularization_loss

        self.optimizer.step()

        return loss, regularization_loss

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

    @helper_method
    def check_flags(self, *args):
        for pairs in args:
            flag_name, flag_value, object_name, object_value = pairs
            if flag_value:
                if object_value is None:
                    print(
                        f"""
\nWARNING: Flag \"{flag_name}\" is enabled, yet the train loader doesn't have a \"{object_name}\" object.
          Consider disabling the flag or passing a {object_name}() instance as a train loader into Train.__init__().\n"""
                    )
