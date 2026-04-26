from ..lib import system
from ..lib import json_engine
from ..lib import pickle_engine
from ..lib import csv_engine


class Serialization:
    @classmethod
    def save_to_npz(cls, backend, filename=None, compressed=False, **kwargs):
        xp = backend.xp

        npz_path = cls.Paths.npz_path
        if filename is None:
            filename = cls.Paths.default_data

        npz_extension = cls.Extensions.npz
        if not filename.endswith(npz_extension):
            filename = filename + npz_extension

        npz_file_path = npz_path
        if npz_file_path is not None and not system.path.exists(npz_file_path):
            system.mkdir(npz_file_path)

        filepath = npz_file_path + filename

        if compressed:
            xp.savez_compressed(filepath, **kwargs)
        else:
            xp.savez(filepath, **kwargs)

    @classmethod
    def save_to_csv(cls, X, Y, filename):
        if filename is None:
            filename = cls.TrainResults().default_data
        if not filename.endswith(cls.Extensions.csv):
            filename = filename + cls.Extensions.csv
        csv_file_path = cls.TrainResults().csv_path
        if csv_file_path is not None and not system.path.exists(csv_file_path):
            system.mkdir(csv_file_path)

        filepath = csv_file_path + filename

        n_features = X.shape[1]
        header = [f"f{i}" for i in range(n_features)] + ["label"]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv_engine.writer(f)
            writer.writerow(header)

            for x_row, y_value in zip(X, Y):
                writer.writerow(list(map(float, x_row)) + [int(y_value)])
        print(f"\nSaved data to CSV at : {filepath}\n")

    @classmethod
    def split_to_records(cls, X, Y):
        records = []

        for x_row, y_value in zip(X, Y):
            records.append({"features": x_row.tolist(), "label": int(y_value)})
        return records

    @classmethod
    def records_to_split(cls, records, backend):
        xp = backend.xp

        X = xp.array([r["features"] for r in records], dtype=xp.float32)
        Y = xp.array([r["label"] for r in records], dtype=xp.int64)
        return X, Y

    @classmethod
    def save_to_json(cls, dataset, filename=None, _encoding=None):
        if filename is None:
            filename = cls.TrainResults().default_data
        if not filename.endswith(cls.Extensions.json):
            filename = filename + cls.Extensions.json
        json_file_path = cls.TrainResults().json_path
        if json_file_path is not None and not system.path.exists(json_file_path):
            system.mkdir(json_file_path)

        filepath = json_file_path + filename

        if _encoding is None:
            _encoding = cls.Encodings().UTF_8

        json_data = {
            "train": cls.split_to_records(dataset["X_train"], dataset["Y_train"]),
            "valid": cls.split_to_records(dataset["X_valid"], dataset["Y_valid"]),
            "test": cls.split_to_records(dataset["X_test"], dataset["Y_test"]),
        }

        with open(filepath, "w", encoding=_encoding) as f:
            json_engine.dump(json_data, f)
        print(f"\nSaved data to JSON at : {filepath}\n")

    @classmethod
    def save_to_pickle(cls, dataset, filename=None):
        if filename is None:
            filename = cls.TrainResults().default_data
        if not filename.endswith(cls.Extensions.pickle):
            filename = filename + cls.Extensions.pickle
        pickle_file_path = cls.TrainResults().pickle_path
        if pickle_file_path is not None and not system.path.exists(pickle_file_path):
            system.mkdir(pickle_file_path)

        filepath = pickle_file_path + filename

        with open(filepath, "wb") as f:
            pickle_engine.dump(dataset, f)
        print(f"\nSaved data to Pickle at : {pickle_file_path}\n")



    
