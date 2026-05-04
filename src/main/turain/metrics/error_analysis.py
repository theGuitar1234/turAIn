from turain.utilities import constant, extension, path
from turain.utilities.annotation import helper_method
from turain.utilities.config import TrainDefaults
from ..lib import system, date_time_engine


class ErrorAnalysis:

    @classmethod
    def error_analysis(
        cls,
        backend,
        confusion_matrix,
        log_error_analysis=False,
        error_analysis_file=None,
        error_analysis_path=None,
        config=None,
    ):
        if config is None:
            config = TrainDefaults()
        epsilon = config.epsilon

        xp = backend.xp

        TP = confusion_matrix[constant.TP]
        TN = confusion_matrix[constant.TN]
        FP = confusion_matrix[constant.FP]
        FN = confusion_matrix[constant.FN]

        total = sum(confusion_matrix.values()) + epsilon

        accuracy = (TP + TN) / total
        percision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        specifity = TN / (TN + FP + epsilon)
        matthews_correlation_coefficient = ((TP * TN) - (FP * FN)) / (
            xp.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + epsilon
        )
        jaccart_index = TP / (TP + FP + FN + epsilon)

        false_positive_rate = 1 - specifity  # FP / (FP + TN)
        false_negative_rate = 1 - recall  # FN / (FN + TP)
        true_negative_rate = specifity
        true_positive_rate = recall
        balanced_accuracy = (recall + specifity) / 2
        f1_score = 2 * (percision * recall) / (percision + recall + epsilon)

        tp, fp, tn, fn, f1, acc, per, rec, spec, fpr, fnr, tnr, tpr, mcc, iou, bal_acc = (
            constant.tp,
            constant.fp,
            constant.tn,
            constant.fn,
            constant.f1,
            constant.acc,
            constant.per,
            constant.rec,
            constant.spec,
            constant.fpr,
            constant.fnr,
            constant.tnr,
            constant.tpr,
            constant.mcc,
            constant.iou,
            constant.bal_acc,
        )

        if log_error_analysis:
            error_analysis_log = {
                tp: TP,
                fp: TN,
                tn: FP,
                fn: FN,
                acc: accuracy,
                per: percision,
                rec: recall,
                f1: f1_score,
                spec: specifity,
                fpr: false_positive_rate,
                fnr: false_negative_rate,
                tnr: true_negative_rate,
                tpr: true_positive_rate,
                mcc: matthews_correlation_coefficient,
                iou: jaccart_index,
                bal_acc: balanced_accuracy,
            }
            cls.log_error_analysis(error_analysis_log, error_analysis_file, error_analysis_path)

        return {
            tp: TP,
            fp: FP,
            tn: TN,
            fn: FN,
            acc: accuracy,
            per: percision,
            rec: recall,
            f1: f1_score,
            spec: specifity,
            fpr: false_positive_rate,
            fnr: false_negative_rate,
            tnr: true_negative_rate,
            tpr: true_positive_rate,
            mcc: matthews_correlation_coefficient,
            iou: jaccart_index,
            bal_acc: balanced_accuracy,
        }

    @helper_method
    @staticmethod
    def log_error_analysis(error_analysis_log, error_analysis_file, error_analysis_path):
        for key, value in error_analysis_log.items():
            print(f"\n{key} : {value}\n")
        if error_analysis_file is None:
            error_analysis_file = constant.ERROR_ANALYSIS_FILENAME
            error_analysis_path = path.ERROR_ANALYSIS_DIRECTORY
        if not system.path.exists(error_analysis_path):
            system.mkdir(error_analysis_path)
        error_analysis_full_path = (
            error_analysis_path + error_analysis_file + extension.TEXT_EXTENSION
        )
        with open(error_analysis_full_path, "a") as f:
            f.write(
                f"\n\nModel Train Error Analysis for date : {date_time_engine.utcnow().isoformat() + "Z"}\n\n"
            )
            f.write(str(error_analysis_log))
        print(f"\nError Analysis Are written to : {error_analysis_full_path}\n")
