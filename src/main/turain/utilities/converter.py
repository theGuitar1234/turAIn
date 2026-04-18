import copy


def scalar_to_python(x, backend, on_gpu):
    xp = backend.xp

    if on_gpu:
        return float(xp.asnumpy(x))
    return float(x)


def to_cpu_WB(backend, on_gpu):
    xp = backend.xp
    if not on_gpu:
        return self.__WB
    return [(xp.asnumpy(W), xp.asnumpy(b)) for W, b in self.__WB]


def cpu_to_cuda_WB(WB, backend, on_gpu):
    xp = backend.xp

    if on_gpu:
        return [(xp.asarray(W), xp.asarray(b)) for W, b in WB]
    return [(xp.asarray(W), xp.asarray(b)) for W, b in WB]


def to_device(x, on_gpu, backend, dtype=None):
    xp = backend.xp

    if on_gpu:
        return xp.asarray(x, dtype=dtype)
    return xp.asarray(x, dtype=dtype)


def to_cpu(x, backend, on_gpu):
    xp = backend.xp

    if on_gpu:
        return xp.asnumpy(x)
    return xp.asarray(x)


def cpu_copy(self):
    model_cpu = copy.copy(self)
    model_cpu.on_gpu = False
    model_cpu._NeuralNetwork__WB = [(W.copy(), b.copy()) for W, b in self.to_cpu_WB()]
    model_cpu._NeuralNetwork__init_random_range = None
    model_cpu._NeuralNetwork__cache = []
    return model_cpu


def move_to(self, device):
    match device:
        case self.Device.CPU:
            wb_cpu = self.to_cpu_WB()
            self._NeuralNetwork__WB = wb_cpu
            self.on_gpu = False
        case self.Device.CUDA:
            if cp is None:
                raise RuntimeError("CuPy is not installed")
            self._NeuralNetwork__WB = [(cp.asarray(W), cp.asarray(b)) for W, b in self.to_cpu_WB()]
            self.on_gpu = True
        case _:
            raise ValueError("Unsupported device")
