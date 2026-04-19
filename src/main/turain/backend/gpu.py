from main.turain.backend.backend import Backend
from lib import gpu_engine
from lib import override_from_parent

class Cupy(Backend):
    gp = gpu_engine
    
    @override_from_parent
    def array(self, x, _dtype=None):
        return self.gp.asarray(x, dtype=_dtype)

    @override_from_parent
    def ones(self, shape, _dtype=None):
        return self.gp.ones(shape, dtype=_dtype)

    @override_from_parent
    def identity(self, size):
        return self.gp.eye(size)

    @override_from_parent
    def uniform_distribution(self, low, high, _size):
        return self.gp.random.uniform(low, high, size=_size)

    @override_from_parent
    def normal_distribution(self, size):
        return self.gp.random.standard_normal(size)

    @override_from_parent
    def sum(self, x, _axis=None, _keepdims=False):
        return self.gp.sum(x, axi=_axis, keepdims=_keepdims)

    @override_from_parent
    def max(self, x, _axis=None, _keepdims=False):
        return self.gp.max(x, axis=_axis, keepdims=_keepdims)

    @override_from_parent
    def e_to_the_power(self, x):
        return self.gp.exp(x)
    
    @override_from_parent
    def hiperbolic_tangent(self, x):
        return self.gp.tanh(x)

    @override_from_parent
    def square_root(self, x):
        return self.gp.sqrt(x)

    @override_from_parent
    def argument_max(self, x, _axis=None):
        return self.gp.argmax(x, axis=_axis)

    @override_from_parent
    def matrix_multiplication(self, a, b):
        return self.gp.matmul(a, b)
    
    @override_from_parent
    def transpoze(self, a):
        return self.gp.transpose(a)
