from lib import date_time_engine
from lib import pickle_engine
from lib import system


class ModeIO:
    @classmethod
    def extract_metadata(cls, model, format_version=None, train_history=None):
        if format_version is None:
            format_version = model.TrainDefaults().default_format_version

        meta = {
            "format_version": format_version,
            "saved_at": date_time_engine.utcnow().isoformat() + "Z",
        }

        if train_history is not None:
            meta["train_history"] = train_history
        return meta

    @staticmethod
    def extract_state(model):
        state = {
            "number_of_layers": model.__L,
            "hidden_activation_type": model.__hidden_activation_type.name,
            "output_activation_type": model.__output_activation_type.name,
            "loss_type": model.__loss_type.name,
            "parameter_count": model.count_parameters(),
        }
        return state

    @staticmethod
    def load_state(model, state):
        for layer, layer_state in zip(model.layers, state):
            parameters = layer.parameters()
            saved_parameters = layer_state["parameters"]

            for parameter, saved_data in zip(parameters, saved_parameters):
                parameter.data = saved_data

    @classmethod
    def save_model(cls, model, file_name, meta=False, format_version=None, train_history=False):
        if not file_name.endswith(".pkl"):
            file_name = file_name + ".pkl"

        modelpath = cls.Paths.model_path
        if modelpath is not None and not system.path.exists(modelpath):
            system.mkdir(modelpath)

        model_cpu = cls.cpu_copy()

        file_path = modelpath + file_name
        with open(file_path, "wb") as f:
            pickle_engine.dump(model_cpu, f)
        print(f"\nSaved the Model to : {file_path}\n")

        if meta:
            meta_data = cls.extract_metadata(format_version, train_history)
            meta_data_path = modelpath + file_name + Paths.meta_data_flair
            with open(meta_data_path, "wb") as f:
                pickle_engine.dump(meta_data, f)
            print(f"\nSaved Meta Data at : {meta_data_path}\n")

    @classmethod
    def load_model(cls, model_path, meta_data_path, device=None, meta=False):
        with open(model_path, "rb") as f:
            model = pickle_engine.load(f)
        print(f"Loaded Model from : {model_path}\n")
        if device is not None:
            model.move_to(device)
        meta_data = None
        if meta:
            with open(meta_data_path, "rb") as f:
                meta_data = pickle_engine.load(f)
            print(f"Loaded Meta data from : {meta_data_path}\n")
        return model, meta_data



    
