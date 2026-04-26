from main.turain.backend.cpu import CPU
from main.turain.neural_network.losses.mean_squared_error import MeanSquaredError
from main.turain.train.finalizer import Finalizer
from main.turain.train.evaluator import Evaluator
from main.turain.metrics.prediction_logger import PredictionLogger
from main.turain.metrics.confusion_matrix import ConfusionMatrix
from main.turain.metrics.error_analysis import ErrorAnalysis

model = None
X_train = None
Y_train = None
X_valid = None
Y_valid = None
X_test = None
Y_test = None

loss_function = MeanSquaredError()
backend = CPU()

evaluator = Evaluator(model, backend)
finalizer = Finalizer(
    evaluator=evaluator,
    prediction_logger=PredictionLogger(),
    confusion_matrix_builder=ConfusionMatrix,
    error_analyzer=ErrorAnalysis,
)

report = finalizer.finalize(
    X_train=X_train,
    Y_train=Y_train,
    X_valid=X_valid,
    Y_valid=Y_valid,
    X_test=X_test,
    Y_test=Y_test,
    loss_function=loss_function,
    threshold=0.5,
    log_predictions=False,
    run_error_analysis=True,
)

print(report.train_loss, report.train_accuracy)
print(report.validation_loss, report.validation_accuracy)
print(report.test_loss, report.test_accuracy)
print(report.error_analysis)