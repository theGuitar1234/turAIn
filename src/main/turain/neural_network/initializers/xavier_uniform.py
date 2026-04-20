class XavierUniform:
    def __call__(self, fan_in, fan_out, rng, xp):
        limit = xp.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_out, fan_in))
