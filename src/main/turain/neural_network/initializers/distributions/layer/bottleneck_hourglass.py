class BottleNeckHourGlass:
    def __call__(self, start_width, output_width, number_of_hidden_layers):
        bottleneck = max(1, min(output_width * 2, start_width // 4))
        if number_of_hidden_layers == 1:
            layers = [bottleneck]
        else:
            left_count = number_of_hidden_layers // 2
            right_count = number_of_hidden_layers - left_count
            left = []
            if left_count > 0:
                left_ratio = (bottleneck / start_width) ** (1 / left_count)
                left = [
                    max(1, int(round(start_width * (left_ratio**l))))
                    for l in range(1, left_count + 1)
                ]
            right = []
            if right_count > 0:
                right_start = left[-1] if left else bottleneck
                right_ratio = (start_width / right_start) ** (1 / right_count)
                right = [
                    max(1, int(round(right_start * (right_ratio**l))))
                    for l in range(1, right_count + 1)
                ]
            layers = (left + right)[:number_of_hidden_layers]
        layers.append(output_width)
        return layers
