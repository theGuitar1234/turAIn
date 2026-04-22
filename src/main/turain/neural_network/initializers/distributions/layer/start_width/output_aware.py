class OutputAware:
    def __call__(self, output_width, input_width, start_width_heuristic_cap):
        return max(output_width * 4, min(start_width_heuristic_cap, input_width))