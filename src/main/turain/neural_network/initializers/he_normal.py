class HeNormal:
    def __call__(self, fan_in, fan_out, alpha, rng, xp):
        standart_deviation = xp.sqrt(2.0 / ((1.0 + alpha**2) * fan_in))
        return rng.standard_normal((fan_out, fan_in)) * standart_deviation