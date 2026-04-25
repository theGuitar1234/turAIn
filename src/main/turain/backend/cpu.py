from .backend import Backend
from lib import cpu_engine
from lib import override_from_parent


class CPU(Backend):
    xp = cpu_engine
    
    def __init__(self):
        if self.xp is None:
            raise RuntimeError("NumPy is not installed")
    
    @override_from_parent
    def to_gpu(self, x):
        return self.xp.ascupy(x)


if __name__ == "__main__":
    pass
