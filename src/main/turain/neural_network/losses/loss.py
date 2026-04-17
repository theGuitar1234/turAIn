class Loss:
    def forward_propagation(self, prediction, true_label):
        raise NotImplementedError

    def backward_propagation(self):
        raise NotImplementedError
