from .backend import Backend
from lib import gpu_engine
from lib import override_from_parent


class Cupy(Backend):
    xp = gpu_engine

    def __init__(self):
        if self.xp is None:
            raise RuntimeError("CuPy is not installed")

    @override_from_parent
    def to_cpu(self, x):
        return self.xp.asnumpy(x)
