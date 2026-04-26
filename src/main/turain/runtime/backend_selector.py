from turain.backend.cpu import CPU
from turain.backend.gpu import GPU
from ..utilities import Device

class BackendSelector:
    @staticmethod
    def select(device_type):
        match device_type:
            case Device.CPU:
                return CPU()
            case Device.CUDA:
                return GPU()
            case _:
                raise ValueError("Unsupported device type")