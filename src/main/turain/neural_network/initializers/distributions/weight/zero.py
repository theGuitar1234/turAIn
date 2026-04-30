class Zero:
    def __call__(fan_in, fan_out, xp):
        return xp.zeros((fan_out, fan_in))
