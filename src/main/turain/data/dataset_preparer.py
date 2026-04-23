from utilities import core_method
from encoding import OneHotCoding

class DatasetPreparer:
    @staticmethod
    @core_method
    def prepare_datasets(
        xp,
        number_of_classes,
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        X_test,
        Y_test,
        one_hot=False,
        flatten_features=False,
        cast_labels_to_int=False,
    ):
        def prepare_features(X):
            if X is None:
                return None
            X = xp.asarray(X)
            if flatten_features and X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            return X
        
        def prepare_labels(Y):
            if Y is None:
                return None
            Y = xp.asarray(Y)
            if cast_labels_to_int:
                Y = Y.astype(xp.int64)
            if one_hot:
                if number_of_classes is None:
                    raise ValueError("number_of_classes must be provided when one_hot=True")
                Y = OneHotCoding.one_hot_decode(Y, number_of_classes)
            return Y

        return {
            "X_train": prepare_features(X_train),
            "Y_train": prepare_labels(Y_train),
            "X_valid": prepare_features(X_valid),
            "Y_valid": prepare_labels(Y_valid),
            "X_test": prepare_features(X_test),
            "Y_test": prepare_labels(Y_test),
        }
