class BiasConstant:
    def __call__(self, xp, fan_out, bias_value):
        return xp.full((fan_out, 1), bias_value)