class HeUniform:
    def __call__(self, alpha, fan_in, fan_out, rng, xp):
        limit = xp.sqrt(6.0 / ((1.0 + alpha**2) * fan_in))
        return rng.uniform(-limit, limit, size=(fan_out, fan_in))
