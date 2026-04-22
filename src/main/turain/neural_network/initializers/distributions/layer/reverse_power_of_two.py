class ReversePowerOfTwo:
    def __call__(self, start_width, output_width, number_of_hidden_layers):
        base_power = max(1, 2 ** int(np.floor(np.log2(start_width))))
        layers = [max(1, base_power * (2**l)) for l in range(number_of_hidden_layers)]
        layers.append(output_width)
        return layers
