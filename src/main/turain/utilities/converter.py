from ..lib import clone
from ..utilities import Device

def scalar_to_python(x, backend, on_gpu):
    if on_gpu:
        return float(backend.asnumpy(x))
    return float(x)


def to_cpu_WB(backend, on_gpu, WB):
    if not on_gpu:
        return WB
    return [(backend.asnumpy(W), backend.asnumpy(b)) for W, b in WB]


def cpu_to_cuda_WB(WB, backend, on_gpu):
    if on_gpu:
        return [(backend.asarray(W), backend.asarray(b)) for W, b in WB]
    return [(backend.asarray(W), backend.asarray(b)) for W, b in WB]


def to_device(x, on_gpu, backend, dtype=None):
    if on_gpu:
        return backend.asarray(x, dtype=dtype)
    return backend.asarray(x, dtype=dtype)


def to_cpu(x, backend, on_gpu):
    if on_gpu:
        return backend.asnumpy(x)
    return backend.asarray(x)


def cpu_copy(model):
    model_cpu = clone.copy(model)
    model_cpu.on_gpu = False
    model_cpu._NeuralNetwork__WB = [(W.copy(), b.copy()) for W, b in to_cpu_WB()]
    model_cpu._NeuralNetwork__init_random_range = None
    model_cpu._NeuralNetwork__cache = []
    return model_cpu


def move_to(device, backend, on_gpu, WB):
    match device:
        case Device.CPU:
            wb_cpu = to_cpu_WB(backend, on_gpu, WB)
            backend._NeuralNetwork__WB = wb_cpu
            backend.on_gpu = False
        case Device.CUDA:
            if backend is None:
                raise RuntimeError("CuPy is not installed")
            backend.WB = [(backend.asarray(W), backend.asarray(b)) for W, b in backend.to_cpu_WB()]
            backend.on_gpu = True
        case _:
            raise ValueError("Unsupported device")
