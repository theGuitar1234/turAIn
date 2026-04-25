class BiasUniform:
    def __call__(self, bias_value, fan_out, rng):
        return rng.uniform(-bias_value, bias_value, size=(fan_out, 1))