from utilities import core_method
from utilities import check_arguments


class Initializer:
    def __init__(self, backend, output_width):
        self.backend = backend
        check_arguments(
            key=int,
            value=(
                output_width,
                {
                    "predicate": lambda output_width: output_width < 1,
                    "error_message": "output_width must be a positive integer",
                },
            )
        )
        self.output_width = output_width


    @core_method
    def initialize(self):
        raise NotImplementedError
