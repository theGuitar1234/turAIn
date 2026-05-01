from .backend import Backend
from ..lib import gpu_engine
from ..lib import cpu_engine
from ..lib import override_from_parent


class GPU(Backend):
    xp = gpu_engine

    def __init__(self):
        if self.xp is None:
            raise RuntimeError("CuPy is not installed")

    @override_from_parent
    def to_cpu(self, x):
        return gpu_engine.asnumpy(x)

    @override_from_parent
    def to_gpu(self, x):
        return cpu_engine.ascupy(x)
