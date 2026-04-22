class ConstantWidth:
    def __call__(self, start_width, number_of_hidden_layers):
        return [start_width] * number_of_hidden_layers
