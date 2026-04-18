from main.turain.backend.backend import Backend
from lib import cpu_engine
from lib import override_from_parent


class CPU(Backend):
    cp = cpu_engine

    @override_from_parent
    def array(self, x, _dtype=None):
        return self.cp.asarray(x, dtype=_dtype)

    @override_from_parent
    def ones(self, shape, _dtype=None):
        return self.cp.ones(shape, dtype=_dtype)

    @override_from_parent
    def identity(self, size):
        return self.cp.eye(size)

    @override_from_parent
    def uniform_distribution(self, low, high, _size):
        return self.cp.random.uniform(low, high, size=_size)

    @override_from_parent
    def normal_distribution(self, size):
        return self.cp.random.standard_normal(size)

    @override_from_parent
    def sum(self, x, _axis=None, _keepdims=False):
        return self.cp.sum(x, axi=_axis, keepdims=_keepdims)

    @override_from_parent
    def max(self, x, _axis=None, _keepdims=False):
        return self.cp.max(x, axis=_axis, keepdims=_keepdims)

    @override_from_parent
    def e_to_the_power(self, x):
        return self.cp.exp(x)

    @override_from_parent
    def square_root(self, x):
        return self.cp.sqrt(x)

    @override_from_parent
    def argument_max(self, x, _axis=None):
        return self.cp.argmax(x, axis=_axis)

    @override_from_parent
    def matrix_multiplication(self, a, b):
        return self.cp.matmul(a, b)
    
    @override_from_parent
    def transpoze(self, a):
        return self.cp.transpose(a)
