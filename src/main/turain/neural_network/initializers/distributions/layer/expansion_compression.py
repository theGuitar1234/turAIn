class ExpansionCompression:
    def __call__(self, start_width, input_width, expansion_multiplier, number_of_hidden_layers, output_width):
        peak_width = max(start_width, input_width * expansion_multiplier)
        if number_of_hidden_layers == 1:
            layers = [peak_width]
        else:
            down_ratio = (output_width / peak_width) ** (1 / number_of_hidden_layers)
            layers = [
                max(1, int(round(peak_width * (down_ratio**l))))
                for l in range(number_of_hidden_layers)
            ]
        layers.append(output_width)
        return layers
