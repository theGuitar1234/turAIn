class LeCunNormal:
    def __call__(self, fan_in, fan_out, rng, xp):
        standart_deviation = xp.sqrt(1.0 / fan_in)
        return rng.standard_normal((fan_out, fan_in)) * standart_deviation