class ParameterBudget:
    def __call__(self, input_width, output_width, number_of_hidden_layers, parameter_budget):
        if parameter_budget < output_width:
            raise ValueError("parameter_budget must be >= output_dim")
        if number_of_hidden_layers == 1:
            width = max(
                1, int((parameter_budget - output_width) / (input_width + output_width + 1))
            )
        else:
            a = number_of_hidden_layers - 1
            b = input_width + output_width + number_of_hidden_layers
            c = output_width - parameter_budget
            disc = max(0.0, b * b - 4 * a * c)
            if a == 0:
                width = max(1, int(-c / max(1, b)))
            else:
                width = max(1, int((-b + np.sqrt(disc)) / (2 * a)))
        layers = [width] * number_of_hidden_layers
        layers.append(output_width)
        return layers
