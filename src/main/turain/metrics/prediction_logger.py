from ..lib import system


class PredictionLogger:
    def log_predictions(
        self,
        Y,
        xp,
        predictions,
        _log_predictions,
        prediction_file,
        prediction_path,
        prediction_tolerance,
        prediction_threshold,
        _encoding,
    ):
        if prediction_path is not None and not system.path.exists(prediction_path):
            system.mkdir(prediction_path)

        if _log_predictions:
            prediction_file_name = prediction_file + self.Extensions.text
            prediction_file_path = prediction_path + prediction_file_name

            if _encoding is None:
                _encoding = self.Encodings().UTF_8

            with open(prediction_file_path, "w", encoding=_encoding) as f:
                for i in range(len(predictions)):
                    sample = Y[i]
                    prediction = predictions[i]
                    sample_result = xp.where(sample == 1)[0]
                    prediction_result = xp.where(prediction == 1)[0]
                    f.write(
                        f"Sample : {sample_result}, Prediction : {prediction_result} {"correct" if sample_result == prediction_result else "failed"}\n"
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
