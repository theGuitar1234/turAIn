class CappedInputWidth:
    def __call__(self, start_width_heuristic_cap, input_width):
        return min(start_width_heuristic_cap, input_width)