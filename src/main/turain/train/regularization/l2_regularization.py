from ...utilities import core_method


class L2Regularization:
    def __init__(self, l2_lambda, backend):
        self.l2_lambda = l2_lambda
        self.backend = backend

    @core_method
    def penalty(self, model, sample_count):
        xp = self.backend.xp

        weight_square_sum = 0.0
        for parameter in model.parameters():
            if self._is_bias(parameter):
                continue
            weight_square_sum += xp.sum(parameter.data * parameter.data)

        return (self.l2_lambda / (2.0 * sample_count)) * weight_square_sum

    @core_method
    def apply_gradient(self, model, sample_count):
        for parameter in model.parameters():
            if parameter.gradient is None or self._is_bias(parameter):
                continue
            parameter.gradient += self.sum_weight_squares(parameter, sample_count)

    def sum_weight_squares(self, parameter, sample_count):
        return (self.l2_lambda / sample_count) * parameter.data

    def _is_bias(self, parameter):
        return getattr(parameter, "is_bias", False) or getattr(parameter, "name", None) == "bias"
