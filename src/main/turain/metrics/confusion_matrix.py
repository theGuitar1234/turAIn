from ..lib import date_time_engine
from ..lib import system
from ..utilities import path
from ..utilities import constant


class ConfusionMatrix:

    @classmethod
    def confusion_matrix(
        cls,
        backend,
        OvR,
        true_labels,
        predicted_labels,
        number_of_classes,
        log_confusion_matrix=False,
        confusion_matrix_file=None,
        confusion_matrix_path=None,
    ):
        if log_confusion_matrix:
            cls.log_confusion_matrix(OvR, number_of_classes, confusion_matrix_file, confusion_matrix_path)

        return cls.one_vs_rest(OvR, true_labels, predicted_labels, backend)

    @classmethod
    def one_vs_rest(cls, target_class, true_labels, predicted_labels, backend):
        xp = backend.xp

        true_labels = xp.asarray(true_labels)
        predicted_labels = xp.asarray(predicted_labels)

        tp = xp.sum((predicted_labels == target_class) & (true_labels == target_class))
        tn = xp.sum((predicted_labels != target_class) & (true_labels != target_class))
        fp = xp.sum((predicted_labels == target_class) & (true_labels != target_class))
        fn = xp.sum((predicted_labels != target_class) & (true_labels == target_class))

        return {
            constant.TP: int(tp),
            constant.TN: int(tn),
            constant.FP: int(fp),
            constant.FN: int(fn),
        }

    @staticmethod
    def log_confusion_matrix(
        OvR, 
        number_of_classes, 
        confusion_matrix_file=None, 
        confusion_matrix_path=None
    ):
        confusion_matrix_str = ""
        confusion_matrix_list = []

        space = constant.SPACE
        padding = constant.PADDING
        for i in range(number_of_classes):
            row = []
            for j in range(number_of_classes):
                if i == OvR and j == OvR:
                    tp = constant.TP
                    confusion_matrix_str += tp + padding * space
                    row.append(tp)
                elif i != OvR and j != OvR:
                    tn = constant.TN
                    confusion_matrix_str += tn + padding * space
                    row.append(tn)
                elif i != OvR and j == OvR:
                    fp = constant.FP
                    confusion_matrix_str += fp + padding * space
                    row.append(fp)
                elif i == OvR and j != OvR:
                    fn = constant.FN
                    confusion_matrix_str += fn + padding * space
                    row.append(fn)
                else:
                    raise ValueError("Invalid Case in log_confusion_matrix(), something went wrong")
            confusion_matrix_str += "\n"
            confusion_matrix_list.append(row)

        _confustion_matrix_log = f"""Confusion Matrix for class : {OvR}\n{confusion_matrix_str}\nList Representation for class : {OvR}\n{confusion_matrix_list}"""
        print()
        print(f"\n{_confustion_matrix_log}\n")
        print()

        if confusion_matrix_file is None:
            confusion_matrix_file = constant.DEFAULT_CONFUSION_MATRIX_FILENAME
            confusion_matrix_path = path.CONFUSION_MATRIX_DIRECTORY
        if not system.path.exists(confusion_matrix_path):
            system.mkdir(confusion_matrix_path)
        confusion_matrix_full_path = (
            confusion_matrix_path + confusion_matrix_file + self.Extensions().text
        )
        with open(confusion_matrix_full_path, "a") as f:
            f.write(
                f"\n\nModel Train Confusion Matrices for date : {date_time_engine.utcnow().isoformat() + "Z"}\n\n"
            )
            f.write(_confustion_matrix_log)
        print(f"\nError Analysis Are written to : {confusion_matrix_path}\n")
