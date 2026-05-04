from turain.utilities import encoding, extension
from turain.utilities.annotation import core_method, helper_method
from ..lib import system


class PredictionLogger:

    @helper_method
    @staticmethod
    def assist_terminance():
        print("\nThe current dataset is too large, >100 samples is expected to be written!")
        print(
            "\nDo you really want to write down all those predictions? If not, input [N/No], otherwise [Y/Yes] (case-insensitive)"
        )
        while True:
            choice = input("Do you wish to continue? [y(Yes)/n(No)] ").strip().lower()
            if choice in ("y", "yes"):
                break
            elif choice in ("n", "no"):
                break
            else:
                print("Invalid Input, supported values are : [N/n/No, Y/y/Yes]")

    @core_method
    @classmethod
    def main(
        cls,
        Y,
        backend,
        predictions,
        log_predictions,
        prediction_file,
        prediction_path,
        prediction_tolerance,
        prediction_threshold,
        _encoding,
    ):
        xp = backend.xp

        if log_predictions:
            if len(Y) > prediction_tolerance:
                if cls.assist_terminance():
                    return

            if Y is not None and predictions is not None:
                cls.log_predictions(
                    Y,
                    xp,
                    predictions,
                    log_predictions,
                    prediction_file,
                    prediction_path,
                    prediction_tolerance,
                    prediction_threshold,
                    _encoding,
                )

    @staticmethod
    def log_predictions(
        Y,
        backend,
        predictions,
        log_predictions,
        prediction_file,
        prediction_path,
        prediction_threshold,
        _encoding,
    ):
        xp = backend.xp
        if prediction_path is not None and not system.path.exists(prediction_path):
            system.mkdir(prediction_path)

        if log_predictions:
            prediction_file_name = prediction_file + extension.TEXT_EXTENSION
            prediction_file_path = prediction_path + prediction_file_name

            if _encoding is None:
                _encoding = encoding.UTF_8

            with open(prediction_file_path, "w", encoding=_encoding) as f:
                for i in range(len(predictions)):
                    sample = Y[i]
                    prediction = predictions[i]
                    sample_result = xp.where(sample == 1)[0]
                    prediction_result = xp.where(prediction == 1)[0]
                    f.write(
                        f"Sample : {sample_result}, Prediction : {prediction_result} {"correct" if xp.array_equal(sample_result, prediction_result) else "failed"}\n"
                    )
            print(
                f"\nFirst <{prediction_threshold} predictions are written in {prediction_file_path}\n"
            )
        else:
            first_sample, last_sample, first_prediction, last_prediction = (
                Y[0],
                Y[-1],
                predictions[0],
                predictions[-1],
            )
            print(
                "Detailed Prediction Logging disabled, falling back to the first and last sample predictions : "
            )
            print(
                f"First Sample : {xp.where(first_sample == 1)[0]}, Prediction : {xp.where(first_prediction == 1)[0]}"
            )
            print(
                f"Last Sample : {xp.where(last_sample == 1)[0]}, Prediction : {xp.where(last_prediction == 1)[0]}\n"
            )
