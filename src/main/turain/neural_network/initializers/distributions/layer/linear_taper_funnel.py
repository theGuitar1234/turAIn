class LinearTaperFunnel:
    def __call__(self, start_width, number_of_hidden_layers, hidden_width):
        return [
            max(
                1,
                int(
                    round(
                        start_width
                        + (l / (number_of_hidden_layers + 1)) * (hidden_width - start_width)
                    )
                ),
            )
            for l in range(1, number_of_hidden_layers + 1)
        ]
