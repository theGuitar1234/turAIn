from ..utilities import core_method
from ..utilities import Datasets
from .deserialization import DeSerialization

class DatasetManager:
    @staticmethod
    @core_method
    def load_dataset(file_type, file_path):
        match file_type:
            case Datasets.NPZ:
                return DeSerialization.load_from_npz(file_path)
            case Datasets.JSON:
                return DeSerialization.load_from_json(file_path)
            case Datasets.PICKLE:
                return DeSerialization.load_from_pickle(file_path)
            case _:
                raise ValueError(
                    f"Unknown File Type {file_type}, supported ones are : {list(Datasets)}"
                )
