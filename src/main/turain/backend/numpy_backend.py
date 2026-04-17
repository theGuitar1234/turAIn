from src.main.python.backend.backend import Backend
from typing import override
import numpy as np


class Numpy(Backend):

    @override
    def identity(self, size):
        return np.eye(size)
