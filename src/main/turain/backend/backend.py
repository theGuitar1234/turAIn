class Backend:
    xp = None

    def to_cpu(self, x):
        raise NotImplementedError

    def to_gpu(self, x):
        raise NotImplementedError
