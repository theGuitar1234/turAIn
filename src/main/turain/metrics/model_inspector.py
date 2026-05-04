class ModelInspector:

    @staticmethod
    def count_parameters(model):
        number_of_parameters = 0
        for W, b in model.parameters():
            number_of_parameters = W.size + b.size
        return number_of_parameters

    @staticmethod
    def parameters_breakdown(model):
        breakdown = []
        for i, (W, b) in enumerate(model.__WB, start=1):
            total_parameters = W.size + b.size
            breakdown.append(
                {
                    "layer": i,
                    "weight_shape": W.shape,
                    "bias_shape": b.shape,
                    "weight_parameters": W.size,
                    "bias_parameters": b.size,
                    "total_parameters": total_parameters,
                }
            )
        return breakdown
