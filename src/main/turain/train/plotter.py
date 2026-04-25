from lib import plotting


class Plotter:
    def __init__(self):
        

    def plot(
        self,
        train_losses,
        val_losses,
        val_accuracies,
        steps,
        epoch,
        train_data_loss,
        validation_data_loss,
        validation_accuracy_loss,
    ):
        train_losses.append(train_data_loss)
        val_losses.append(validation_data_loss)
        val_accuracies.append(validation_accuracy_loss)
        steps.append(epoch)

        plotting.ion()
        plotting.title(self.TrainResults.figure_title)
        plotting.plot(steps, train_losses, label="Train Loss")
        plotting.plot(steps, val_losses, label="Validation Loss")
        plotting.xlabel("iteration")
        plotting.ylabel("loss")
        plotting.pause(0.05)
