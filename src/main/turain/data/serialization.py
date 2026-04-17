import pickle
import json
import csv
import os


def save_model(self, file_name, meta=False, format_version=None, train_history=False):

    if not file_name.endswith(".pkl"):
        file_name = file_name + ".pkl"

    modelpath = self.Paths.model_path
    if modelpath is not None and not os.path.exists(modelpath):
        os.mkdir(modelpath)

    model_cpu = self.cpu_copy()

    file_path = modelpath + file_name
    with open(file_path, "wb") as f:
        pickle.dump(model_cpu, f)
    print(f"\nSaved the Model to : {file_path}\n")

    if meta:
        meta_data = self.get_metadata(format_version, train_history)
        meta_data_path = modelpath + file_name + self.Paths.meta_data_flair
        with open(meta_data_path, "wb") as f:
            pickle.dump(meta_data, f)
        print(f"\nSaved Meta Data at : {meta_data_path}\n")


def save_to_npz(cls, backend, filename=None, compressed=False, **kwargs):
    xp = backend.xp
    
    npz_path = cls.Paths.npz_path
    if filename is None:
        filename = cls.Paths.default_data

    npz_extension = cls.Extensions.npz
    if not filename.endswith(npz_extension):
        filename = filename + npz_extension

    npz_file_path = npz_path
    if npz_file_path is not None and not os.path.exists(npz_file_path):
        os.mkdir(npz_file_path)

    filepath = npz_file_path + filename

    if compressed:
        xp.savez_compressed(filepath, **kwargs)
    else:
        xp.savez(filepath, **kwargs)


def save_to_csv(cls, X, Y, filename):
    if filename is None:
        filename = cls.TrainResults().default_data
    if not filename.endswith(cls.Extensions.csv):
        filename = filename + cls.Extensions.csv
    csv_file_path = cls.TrainResults().csv_path
    if csv_file_path is not None and not os.path.exists(csv_file_path):
        os.mkdir(csv_file_path)

    filepath = csv_file_path + filename

    n_features = X.shape[1]
    header = [f"f{i}" for i in range(n_features)] + ["label"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for x_row, y_value in zip(X, Y):
            writer.writerow(list(map(float, x_row)) + [int(y_value)])
    print(f"\nSaved data to CSV at : {filepath}\n")


def split_to_records(cls, X, Y):
    records = []

    for x_row, y_value in zip(X, Y):
        records.append({"features": x_row.tolist(), "label": int(y_value)})
    return records


def save_to_json(cls, dataset, filename=None, _encoding=None):
    if filename is None:
        filename = cls.TrainResults().default_data
    if not filename.endswith(cls.Extensions.json):
        filename = filename + cls.Extensions.json
    json_file_path = cls.TrainResults().json_path
    if json_file_path is not None and not os.path.exists(json_file_path):
        os.mkdir(json_file_path)

    filepath = json_file_path + filename

    if _encoding is None:
        _encoding = cls.Encodings().UTF_8

    json_data = {
        "train": cls.split_to_records(dataset["X_train"], dataset["Y_train"]),
        "valid": cls.split_to_records(dataset["X_valid"], dataset["Y_valid"]),
        "test": cls.split_to_records(dataset["X_test"], dataset["Y_test"]),
    }

    with open(filepath, "w", encoding=_encoding) as f:
        json.dump(json_data, f)
    print(f"\nSaved data to JSON at : {filepath}\n")


def records_to_split(cls, records, backend):
    xp = backend.xp
    
    X = xp.array([r["features"] for r in records], dtype=xp.float32)
    Y = xp.array([r["label"] for r in records], dtype=xp.int64)
    return X, Y


def save_to_pickle(cls, dataset, filename=None):
    if filename is None:
        filename = cls.TrainResults().default_data
    if not filename.endswith(cls.Extensions.pickle):
        filename = filename + cls.Extensions.pickle
    pickle_file_path = cls.TrainResults().pickle_path
    if pickle_file_path is not None and not os.path.exists(pickle_file_path):
        os.mkdir(pickle_file_path)

    filepath = pickle_file_path + filename

    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)
    print(f"\nSaved data to Pickle at : {pickle_file_path}\n")
