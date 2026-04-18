from datetime import datetime


def get_metadata(self, format_version=None, train_history=None):
    if format_version is None:
        format_version = self.TrainDefaults().default_format_version
    state = {
        "format_version": format_version,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "number_of_layers": self.__L,
        "hidden_activation_type": self.__hidden_activation_type.name,
        "output_activation_type": self.__output_activation_type.name,
        "loss_type": self.__loss_type.name,
        "parameter_count": self.count_parameters(),
    }
    if train_history is not None:
        state["train_history"] = train_history
    return state
