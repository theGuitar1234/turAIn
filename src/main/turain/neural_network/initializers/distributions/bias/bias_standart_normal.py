class BiasStandartNormal:
    def __call__(rng, fan_out, standart_deviation):
        return rng.standard_normal((fan_out, 1)) * standart_deviation