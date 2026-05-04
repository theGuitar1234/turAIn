from turain.utilities.annotation import helper_method
from turain.utilities.config import TrainResults

from ..lib import plotting


class Plotter:

    def plot_real_time(
        self,
        results,
    ):
        epochs = range(1, len(results.train_losses) + 1)

        plotting.plot(epochs, results.validation_losses, label="Validation Loss")
        plotting.plot(epochs, results.train_losses, label="Train Loss")
        plotting.pause(0.05)

    def plot_once(
        self,
        results,
    ):
        plotting.ioff()
        epochs = range(1, len(results.train_losses) + 1)

        plotting.plot(epochs, results.validation_losses, label="Validation Loss")
        plotting.plot(epochs, results.train_losses, label="Train Loss")
        plotting.show()

    @helper_method
    def settle_plot(self, results):
        if plotting is None:
            return None
        plotting.figure()
        plotting.title(results.title if results.title else TrainResults().figure_title)
        plotting.xlabel("iteration")
        plotting.ylabel("loss")
        plotting.legend()
