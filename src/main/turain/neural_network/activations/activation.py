from core.module import Module
from lib import override_from_parent
from utilities import core_method


class Activation(Module):
    def __init__(self, z, backend):
        super().__init__()
        self.backend = backend
        self.input_cache = None
        self.output_cache = None
        self.z = z

    @core_method
    def activate(self, z):
        raise NotImplementedError

    @core_method
    def derivative(self, z):
        raise NotImplementedError
