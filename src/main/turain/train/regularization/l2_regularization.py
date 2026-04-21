from utilities import TrainDefaults
from utilities import core_method


class L2Regularization:
    def __init__(self, backend, cfg=None):
        if cfg is None:
            cfg = TrainDefaults()
        self.l2_lambda = cfg.l2_lambda
        self.backend = backend

    @core_method
    def penalty(self, model, sample_count):
        xp = self.backend

        weight_square_sum = 0.0
        for parameter in model.parameters():
            if hasattr(parameter, "name") and parameter.name == "bias":
                continue
            if getattr(parameter, "is_bias", False):
                continue
            weight_square_sum += xp.sum(parameter.data * parameter.data)
        return (self.l2_lambda / (2.0 * sample_count)) * weight_square_sum

    def sum_weight_squares(self, parameter, sample_count):
        return (self.l2_lambda / sample_count) * parameter.data


if __name__ == "__main__":
    pass
