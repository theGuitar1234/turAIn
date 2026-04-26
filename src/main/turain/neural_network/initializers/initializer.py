from ...utilities import core_method
from ...utilities import check_positive_integer


class Initializer:
    def __init__(self, backend, output_width):
        self.backend = backend
        check_positive_integer(output_width)
        self.output_width = output_width


    @core_method
    def initialize(self):
        raise NotImplementedError
