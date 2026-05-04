from ..utilities import core_method


class Logger:
    def __init__(self, epoch=1):
        self.epoch = epoch

    def should_log(self, epoch):
        return self.epoch is not None and self.epoch > 0 and epoch % self.epoch == 0

    def log_epoch(
        self, epoch, train_loss, validation_loss=None, validation_accuracy=None, learning_rate=None
    ):
        parts = [f"epoch={epoch}", f"train_loss={train_loss:.6f}"]

        if validation_loss is not None:
            parts.append(f"validation_loss={validation_loss:.6f}")

        if validation_accuracy is not None:
            parts.append(f"validation_accuracy={validation_accuracy:.4f}")

        if learning_rate is not None:
            parts.append(f"learning_rate={learning_rate}")

        print(f"[ {" ][ ".join(parts)}")
