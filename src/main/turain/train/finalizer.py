from turain.metrics.confusion_matrix import ConfusionMatrix
from turain.metrics.error_analysis import ErrorAnalysis
from turain.metrics.prediction_logger import PredictionLogger
from turain.utilities.config import TrainDefaults

from ..utilities import TrainResults


class Finalizer:
    def __init__(
        self,
        evaluator,
        backend,
        number_of_classes,
    ):
        self.backend = backend
        self.evaluator = evaluator
        self.number_of_classes = number_of_classes

    def finalize(
        self,
        X_train,
        Y_train,
        X_valid=None,
        Y_valid=None,
        X_test=None,
        Y_test=None,
        prediction_tolerance=None,
        prediction_threshold=None,
        run_error_analysis=False,
        log_predictions=False,
        log_error_analysis=False,
        log_confusion_matrix=False,
        prediction_file=None,
        prediction_path=None,
        error_analysis_file=None,
        error_analysis_path=None,
        confusion_matrix_file=None,
        confusion_matrix_path=None,
        encoding=None,
        config=None,
    ):
        if config is None:
            config = TrainDefaults()
        results = TrainResults()

        train_prediction, _, train_loss, train_accuracy = (
            self.evaluator.evaluate(X_train, Y_train, config)
        )
        results.train_loss = train_loss
        results.train_accuracy = train_accuracy
        results.train_prediction = train_prediction

        if X_valid is not None and Y_valid is not None:
            (
                validation_prediction,
                _,
                validation_loss,
                validation_accuracy,
            ) = self.evaluator.evaluate(X_valid, Y_valid, config)
            results.validation_loss = validation_loss
            results.validation_accuracy = validation_accuracy
            results.validation_prediction = validation_prediction

        if X_test is not None and Y_test is not None:
            test_prediction, test_predicted_classes, test_loss, test_accuracy = (
                self.evaluator.evaluate(X_test, Y_test, config)
            )
            results.test_loss = test_loss
            results.test_accuracy = test_accuracy
            results.test_prediction = test_prediction

            if run_error_analysis:
                true_labels = self._decode_labels(Y_test)
                predicted_labels = self._decode_labels(test_predicted_classes)
                # number_of_classes = sorted(set(true_labels.tolist()))

                for target_class in range(self.number_of_classes):
                    confusion_matrix = ConfusionMatrix.confusion_matrix(
                        backend=self.backend,
                        OvR=target_class,
                        true_labels=true_labels,
                        predicted_labels=predicted_labels,
                        log_confusion_matrix=log_confusion_matrix,
                        confusion_matrix_file=confusion_matrix_file,
                        confusion_matrix_path=confusion_matrix_path,
                    )
                    results.error_analysis[target_class] = ErrorAnalysis.error_analysis(
                        backend=self.backend,
                        confusion_matrix=confusion_matrix,
                        log_error_analysis=log_error_analysis,
                        error_analysis_file=error_analysis_file,
                        error_analysis_path=error_analysis_path,
                    )

        if log_predictions and self.prediction_logger is not None:
            if X_test is not None and Y_test is not None:
                PredictionLogger.log_predictions(
                    Y_test,
                    self.backend,
                    test_prediction,
                    log_predictions=False,
                    prediction_file=prediction_file,
                    prediction_path=prediction_path,
                    prediction_tolerance=prediction_tolerance,
                    prediction_threshold=prediction_threshold,
                    encoding=encoding,
                )

        return results

    def _decode_labels(self, Y):
        xp = self.backend.xp

        Y = xp.asarray(Y)
        if Y.ndim == 2 and Y.shape[1] > 1:
            return xp.argmax(Y, axis=1)
        if Y.ndim == 2 and Y.shape[1] == 1:
            return Y.flatten()
        return Y

    @staticmethod
    def log(results):
        print("\nFinal Results:\n")
        print("Train:", results.train_loss, results.train_accuracy)

        if results.validation_loss is not None:
            print("Valid:", results.validation_loss, results.validation_accuracy)

        if results.test_loss is not None:
            print("Test :", results.test_loss, results.test_accuracy)

        if results.error_analysis:
            print("\nError Analysis:\n")
            for target_class, metrics in results.error_analysis.items():
                print(f"Class {target_class}:")
                print(metrics)
                print()
