from utilities import check_arguments


class Module:
    def __init__(
        self,
        input_features,
        output_features,
        backend,
    ):
        check_arguments(
            key=int,
            value=(
                input_features,
                {
                    "predicate": lambda input_features: input_features < 1,
                    "error_message": "input_of_features must be a positive integer",
                },
            ),
            key=int,
            value=(
                output_features,
                {
                    "predicate": lambda output_features: output_features < 1,
                    "error_message": "output_of_features must be a positive integer",
                },
            ),
        )
        
        self.backend = backend
        self.training = True

    def forward_propagation(self, x):
        raise NotImplementedError

    def backward_propagation(self, gradient_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self
