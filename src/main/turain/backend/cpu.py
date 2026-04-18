from main.turain.backend.backend import Backend
from lib import cpu_engine, override

class CPU(Backend):

    @override
    def identity(self, size):
        return cpu_engine.eye(size)
