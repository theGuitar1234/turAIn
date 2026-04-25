class Backend:
    xp = None

    def array(self, x, _dtype=None):
        return self.xp.asarray(x, dtype=_dtype)

    def ones(self, shape, _dtype=None):
        return self.xp.ones(shape, dtype=_dtype)

    def zeros(self, shape, _dtype=None):
        return self.xp.zeros(shape, dtype=_dtype)

    def zeros_like(self, x):
        return self.xp.zeros_like(x)

    def ones_like(self, x):
        return self.xp.ones_like(x)
    
    def empty_like(self, x):
        return self.xp.empty_like(x)

    def identity(self, size):
        return self.xp.eye(size)

    def uniform_distribution(self, low, high, _size):
        return self.xp.random.uniform(low, high, size=_size)

    def normal_distribution(self, size):
        return self.xp.random.standard_normal(size)

    def sum(self, x, _axis=None, _keepdims=False):
        return self.xp.sum(x, axi=_axis, keepdims=_keepdims)

    def max(self, x, _axis=None, _keepdims=False):
        return self.xp.max(x, axis=_axis, keepdims=_keepdims)

    def exp(self, x):
        return self.xp.exp(x)

    def hiperbolic_tangent(self, x):
        return self.xp.tanh(x)

    def sqrt(self, x):
        return self.xp.sqrt(x)

    def argmax(self, x, _axis=None):
        return self.xp.argmax(x, axis=_axis)

    def clip(self, x, a, b):
        return self.xp.clip(x, a, b)

    def matrix_multiplication(self, a, b):
        return self.xp.matmul(a, b)

    def transpoze(self, a):
        return self.xp.transpose(a)

    def in_range(self, *args, **kwargs):
        return self.xp.arange(*args, **kwargs)

    def to_cpu(self, x):
        raise NotImplementedError

    def to_gpu(self, x):
        raise NotImplementedError
