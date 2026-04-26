from .callback import Callback
from ...lib import override_from_parent


class CallbackManager(Callback):
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    @override_from_parent
    def on_train_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    @override_from_parent
    def on_epoch_begin(self, trainer, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)

    @override_from_parent
    def on_epoch_end(self, trainer, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch)

    @override_from_parent
    def on_train_end(self, trainer):
        for callback in self.callbacks:
            callback.on_train_end(trainer)
