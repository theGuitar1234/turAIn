class Metrics:
    def count_parameters(self):
        number_of_parameters = 0
        for W, b in self.__WB:
            number_of_parameters = W.size + b.size
        return number_of_parameters

    def parameters_breakdown(self):
        breakdown = []
        for i, (W, b) in enumerate(self.__WB, start=1):
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

    def summary(self):
        print("\nNeural Network Summary\n")
        print(f"Number of layers: {self.__L}")
        print(f"Hidden Activation : {self.__hidden_activation_type.name}")
        print(f"Output Activation : {self.__output_activation_type.name}")
        print(f"Loss Type : {self.__loss_type.name}")
        print(f"Total Number of Parameters : {self.count_parameters()}")
        print()
        breakdown = self.parameters_breakdown()
        print(breakdown)


if __name__ == "__main__":
    pass
