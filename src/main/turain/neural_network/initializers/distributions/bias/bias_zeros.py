class BiasZeros:
    def __call__(xp, fan_out):
        return xp.zeros((fan_out, 1))