class LinearTaperFunnelOutput:
    def __call__(self, start_width, number_of_hidden_layers, output_width):
        return [
            max(
                1,
                int(
                    round(
                        start_width
                        + (l / (number_of_hidden_layers + 1)) * (output_width - start_width)
                    )
                ),
            )
            for l in range(1, number_of_hidden_layers + 1)
        ]
