from turain.utilities.config import TrainResults

from ..lib import plotting


class Plotter:
    def plot(
        self,
        results,
        plot_real_time=False,
    ):
        if plotting is None:
            return None
        epochs = range(1, len(results.train_losses) + 1)

        plotting.figure()
        plotting.title(results.title if results.title else TrainResults().figure_title)
        plotting.plot(epochs, results.train_losses, label="Train Loss")

        if plot_real_time:
            if results.validation_losses:
                plotting.plot(epochs, results.train_losses, label="Train Loss")
                plotting.plot(epochs, results.validation_losses, label="Validation Loss")
                plotting.pause(0.05)

        plotting.plot(epochs, results.train_losses, label="Train Loss")
        plotting.plot(epochs, results.validation_losses, label="Validation Loss")

        plotting.xlabel("iteration")
        plotting.ylabel("loss")
        plotting.legend()
        plotting.show()
