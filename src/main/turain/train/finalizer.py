from main.turain.utilities.config import TrainResults


class Finalizer:
    def __init__(
        self,
        evaluator,
        prediction_logger=None,
        confusion_matrix_builder=None,
        error_analyzer=None,
    ):
        self.evaluator = evaluator
        self.prediction_logger = prediction_logger
        self.confusion_matrix_builder = confusion_matrix_builder
        self.error_analyzer = error_analyzer

    def finalize(
        self,
        model,
        X_train,
        Y_train,
        X_valid=None,
        Y_valid=None,
        X_test=None,
        Y_test=None,
        loss_function=None,
        threshold=0.5,
        log_predictions=False,
        prediction_paths=None,
        run_error_analysis=False,
    ):
        report = TrainResults()

        train_result = self.evaluator.evaluate(X_train, Y_train, loss_function, threshold=threshold)
        report.train_loss = train_result["loss"]
        report.train_accuracy = train_result["accuracy"]
        report.train_prediction = train_result["prediction"]

        if X_valid is not None and Y_valid is not None:
            valid_result = self.evaluator.evaluate(
                X_valid, Y_valid, loss_function, threshold=threshold
            )
            report.validation_loss = valid_result["loss"]
            report.validation_accuracy = valid_result["accuracy"]
            report.validation_prediction = valid_result["prediction"]

        if X_test is not None and Y_test is not None:
            test_result = self.evaluator.evaluate(
                X_test, Y_test, loss_function, threshold=threshold
            )
            report.test_loss = test_result["loss"]
            report.test_accuracy = test_result["accuracy"]
            report.test_prediction = test_result["prediction"]

            if (
                run_error_analysis
                and self.confusion_matrix_builder is not None
                and self.error_analyzer is not None
            ):
                true_labels = self._decode_labels(Y_test)
                predicted_labels = self._decode_labels(test_result["predicted_classes"])

                classes = sorted(set(true_labels.tolist()))
                for target_class in classes:
                    confusion = self.confusion_matrix_builder.one_vs_rest(
                        target_class,
                        true_labels,
                        predicted_labels,
                    )
                    report.error_analysis[target_class] = self.error_analyzer.from_confusion(
                        confusion
                    )

        if log_predictions and self.prediction_logger is not None and prediction_paths is not None:
            if "train" in prediction_paths:
                self.prediction_logger.log_predictions(
                    prediction_paths["train"],
                    self._decode_labels(Y_train),
                    self._decode_labels(train_result["predicted_classes"]),
                )

            if X_valid is not None and Y_valid is not None and "valid" in prediction_paths:
                self.prediction_logger.log_predictions(
                    prediction_paths["valid"],
                    self._decode_labels(Y_valid),
                    self._decode_labels(valid_result["predicted_classes"]),
                )

            if X_test is not None and Y_test is not None and "test" in prediction_paths:
                self.prediction_logger.log_predictions(
                    prediction_paths["test"],
                    self._decode_labels(Y_test),
                    self._decode_labels(test_result["predicted_classes"]),
                )

        return report

    def _decode_labels(self, Y):
        import numpy as np

        Y = np.asarray(Y)
        if Y.ndim == 2 and Y.shape[1] > 1:
            return np.argmax(Y, axis=1)
        if Y.ndim == 2 and Y.shape[1] == 1:
            return Y.flatten()
        return Y
    
    @staticmethod
    def print(report):
        print("\nFinal Results:\n")
        print("Train:", report.train_loss, report.train_accuracy)

        if report.validation_loss is not None:
            print("Valid:", report.validation_loss, report.validation_accuracy)

        if report.test_loss is not None:
            print("Test :", report.test_loss, report.test_accuracy)

        if report.error_analysis:
            print("\nError Analysis:\n")
            for target_class, metrics in report.error_analysis.items():
                print(f"Class {target_class}:")
                print(metrics)
                print()
