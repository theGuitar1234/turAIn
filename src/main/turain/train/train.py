class Train:
    def __init__(self, model, loss_function, optimizer, metrics=None, callbacks=None):
        raise NotImplementedError

    def fit(self, train_loader, validation_loader=None, epochs=5000):
        raise NotImplementedError
