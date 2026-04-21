class Zero:
    def __call__(self, fan_in, fan_out, xp):
        return xp.zeros((fan_out, fan_in))
