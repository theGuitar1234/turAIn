class Callback:
    def on_train_begin(self, trainer):
        raise NotImplementedError

    def on_epoch_begin(self, trainer, epoch):
        raise NotImplementedError

    def on_epoch_end(self, trainer, epoch):
        raise NotImplementedError

    def on_train_end(self, trainer):
        raise NotImplementedError