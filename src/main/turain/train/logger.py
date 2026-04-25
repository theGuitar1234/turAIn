from utilities import core_method


class Logger:
    def __init__(self):
        

    @core_method
    def log(
        self,
        epoch,
        train_data_loss,
        train_regularization_loss,
        train_total_loss,
        validation_data_loss,
        validation_accuracy_loss
    ):
        log = []
        print(
            "epoch =",
            epoch,
            "train_data_loss =",
            round(train_data_loss, 6),
            "train_reg_loss =",
            round(train_regularization_loss, 6),
            "train_total_loss =",
            round(train_total_loss, 6),
            "val_data_loss =",
            round(validation_data_loss, 6),
            "val_acc =",
            round(validation_accuracy_loss, 4),
        )
