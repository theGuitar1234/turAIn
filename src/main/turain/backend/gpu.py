from main.turain.backend.backend import Backend
from lib import gpu_engine as gp
from lib import override_from_parent

class Cupy(Backend):
    
    @override_from_parent
    def identity(self, size):
        return gp.eye(size)