class ErrorAnalysis:
    def error_analysis(
        self,
        confusion_matrix,
        _log_error_analysis=False,
        error_analysis_file=None,
        error_analysis_path=None,
    ):
        TP = confusion_matrix[0]
        TN = confusion_matrix[1]
        FP = confusion_matrix[2]
        FN = confusion_matrix[3]

        total = sum(confusion_matrix)

        accuracy = (TP + TN) / total
        percision = TP / (TP + FP)
        recall = TP / (TP + FN)
        specifity = TN / (TP + FP)
        # FP / (FP + TN)
        false_positive_rate = 1 - specifity
        # FN / (FN + TP)
        false_negative_rate = 1 - recall
        true_negative_rate = specifity
        true_positive_rate = recall
        balanced_accuracy = (recall + specifity) / 2
        matthews_correlation_coefficient = ((TP * TN) - (FP * FN)) / (
            np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        )
        jaccart_index = TP / (TP + FP + FN)

        tp = self.ErrorAnalysis().tp
        fp = self.ErrorAnalysis().fp
        tn = self.ErrorAnalysis().tn
        fn = self.ErrorAnalysis().fn
        f1 = self.ErrorAnalysis().f1
        acc = self.ErrorAnalysis().acc
        per = self.ErrorAnalysis().per
        rec = self.ErrorAnalysis().rec
        spec = self.ErrorAnalysis().spec
        fpr = self.ErrorAnalysis().fpr
        fnr = self.ErrorAnalysis().fnr
        tnr = self.ErrorAnalysis().tnr
        tpr = self.ErrorAnalysis().tpr
        mcc = self.ErrorAnalysis().mcc
        iou = self.ErrorAnalysis().iou
        bal_acc = self.ErrorAnalysis().bal_acc

        if _log_error_analysis:
            _error_analysis_log = f"""{tp}: {TP}\n{fp}: {TN}\n{tn}: {FP}\n{fn}: {FN}\n{acc}: {(TP + TN) / total}\n{per}: {percision}\n{rec}: {recall}\n{f1}: {2 * (percision * recall) / (percision + recall)}, \n{spec}: {specifity}, \n{fpr}: {false_positive_rate}, \n{fnr}: {false_negative_rate}, \n{tnr}: {true_negative_rate}, \n{tpr}: {true_positive_rate}, \n{mcc}: {matthews_correlation_coefficient}, \n{iou}: {jaccart_index}, \n{bal_acc}: {balanced_accuracy}"""
            print(f"\n{_error_analysis_log}\n")

            if error_analysis_file is None:
                error_analysis_file = self.Paths().error_analysis_file
                error_analysis_path = self.Paths().error_analysis_path
            if not os.path.exists(error_analysis_path):
                os.mkdir(error_analysis_path)
            error_analysis_full_path = (
                error_analysis_path + error_analysis_file + self.Extensions().text
            )
            with open(error_analysis_full_path, "a") as f:
                f.write(
                    f"\n\nModel Train Error Analysis for date : {datetime.utcnow().isoformat() + "Z"}\n\n"
                )
                f.write(_error_analysis_log)
            print(f"\nError Analysis Are written to : {error_analysis_full_path}\n")

        return {
            tp: TP,
            fp: FP,
            tn: TN,
            fn: FN,
            acc: accuracy,
            per: percision,
            rec: recall,
            f1: 2 * (percision * recall) / (percision + recall),
            spec: specifity,
            fpr: false_positive_rate,
            fnr: false_negative_rate,
            tnr: true_negative_rate,
            tpr: true_positive_rate,
            mcc: matthews_correlation_coefficient,
            iou: jaccart_index,
            bal_acc: balanced_accuracy,
        }
