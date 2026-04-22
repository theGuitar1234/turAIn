class XavierNormal:
    def __call__(self, fan_in, fan_out, xp):
        standart_deviation = xp.square_root(2.0 / (fan_in + fan_out))
        return xp.random.standard_normal((fan_out, fan_in)) * standart_deviation