from main.turain.train.callback import Callback


class PlottingCallback(Callback):
    def __init__(self, plotter):
        self.plotter = plotter

    def on_epoch_end(self, trainer, epoch):
        if hasattr(self.plotter, "update"):
            self.plotter.update(trainer.results)

    def on_train_end(self, trainer):
        self.plotter.plot(trainer.results)
