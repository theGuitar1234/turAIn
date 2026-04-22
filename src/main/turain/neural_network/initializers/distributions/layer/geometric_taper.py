class GeometricTaper:
    def __call__(self, start_width, output_width, number_of_hidden_layers):
        ratio = (output_width / start_width) ** (1 / (number_of_hidden_layers + 1))
        return [
            max(1, int(round(start_width * (ratio**l))))
            for l in range(1, number_of_hidden_layers + 1)
        ]
