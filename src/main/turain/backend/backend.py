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

    def sum(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def max(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def exp(self, x):
        raise NotImplementedError

    def sqrt(self, x, a, b):
        raise NotImplementedError

    def clip(self, x, a, b):
        raise NotImplementedError

    def argmax(self, x, axix=None):
        raise NotImplementedError

    def to_cpu(self, x):
        raise NotImplementedError
