from lib import csv_engine
from lib import json_engine
from lib import pickle_engine
from utilities import Datasets
from utilities import core_method
from serialization import Serialization


class DeSerialization:
    @classmethod
    def load_csv_engine_datasets(cls, folder):
        X_train, Y_train = cls.load_from_csv(f"{folder}/train.csv_engine")
        X_valid, Y_valid = cls.load_from_csv(f"{folder}/valid.csv_engine")
        X_test, Y_test = cls.load_from_csv(f"{folder}/test.csv_engine")

        return {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_valid": X_valid,
            "Y_valid": Y_valid,
            "X_test": X_test,
            "Y_test": Y_test,
        }

    @classmethod
    def load_from_pickle(cls, filepath):
        with open(filepath, "rb") as f:
            dataset = pickle_engine.load(f)
        print(f"\nLoaded data from Pickle at : {filepath}\n")
        return dataset

    @classmethod
    def load_from_json(cls, filepath, _encoding=None):
        if _encoding is None:
            _encoding = cls.Encodings.UTF_8

        with open(filepath, "r", encoding=_encoding) as f:
            data = json_engine.load(f)
        print(f"Loaded data from JSON at : {filepath}\n")

        X_train, Y_train = Serialization.records_to_split(data["train"])
        X_valid, Y_valid = Serialization.records_to_split(data["valid"])
        X_test, Y_test = Serialization.records_to_split(data["test"])

        return {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_valid": X_valid,
            "Y_valid": Y_valid,
            "X_test": X_test,
            "Y_test": Y_test,
        }

    @classmethod
    def load_from_csv_engine(cls, filepath, backend, _newline="", _encoding=None):
        xp = backend.xp

        if _encoding is None:
            _encoding = cls.Encodings().UTF_8
        with open(filepath, "r", newline=_newline, encoding=_encoding) as f:
            reader = csv_engine.reader(f)
            header = next(reader)

            rows = list(reader)
        print(f"\nLoaded a CSV dataset from : {filepath}\n")

        data = xp.array(rows, dtype=xp.float32)

        X = data[:, :-1]
        Y = data[:, -1].astype(xp.int64)

        return X, Y

    @classmethod
    def load_from_npz(cls, npz_path, backend):
        xp = backend.xp

        lib = xp.load(npz_path)

        X_train_3D = lib["X_train"]
        Y_train = lib["Y_train"]

        X_valid_3D = lib["X_valid"]
        Y_valid = lib["Y_valid"]

        X_test_3D = lib["X_test"]
        Y_test = lib["Y_test"]

        X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
        X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
        X_test = X_test_3D.reshape((X_test_3D.shape[0], -1))

        return {
            "X_train": X_train,
            "Y_train": Y_train.astype(xp.int64),
            "X_valid": X_valid,
            "Y_valid": Y_valid.astype(xp.int64),
            "X_test": X_test,
            "Y_test": Y_test.astype(xp.int64),
        }


if __name__ == "__main__":
    pass
