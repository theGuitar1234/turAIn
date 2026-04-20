class LeCunUniform:
    def __call__(self, fan_in, fan_out, rng, xp):
        limit = xp.sqrt(3.0 / fan_in)
        return rng.uniform(-limit, limit, size=(fan_out, fan_in))