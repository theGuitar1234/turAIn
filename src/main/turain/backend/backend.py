class Backend:

    def array(self, x, dtype=None):
        raise NotImplementedError

    def zeros(self, x):
        raise NotImplementedError

    def ones(self, x):
        raise NotImplementedError

    def identity(self, size):
        raise NotImplementedError

    def uniform_distribution(self, low, high, size):
        raise NotImplementedError

    def normal_distribution(self, size):
        raise NotImplementedError

    def matrix_multiplication(self, a, b):
        raise NotImplementedError
    
    def transpoze(self, a):
        raise NotImplementedError

    def sum(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def max(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def e_to_the_power(self, x):
        raise NotImplementedError

    def square_root(self, x, a, b):
        raise NotImplementedError

    def clip(self, x, a, b):
        raise NotImplementedError

    def argument_max(self, x, axis=None):
        raise NotImplementedError

    def to_cpu(self, x):
        raise NotImplementedError
