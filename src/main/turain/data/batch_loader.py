from ..utilities import core_method


class BatchLoader:
    def __init__(self, X, Y, batch_size, backend, shuffle=True, data_augmentation=None):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.backend = backend
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation

    @core_method
    def __iter__(self):
        xp = self.backend.xp
        number_of_samples = self.X.shape[0]

        indices = xp.arange(number_of_samples)

        if self.shuffle:
            xp.random.shuffle(indices)

        X_epoch = self.X
        Y_epoch = self.Y

        if self.data_augmentation is not None:
            X_epoch, Y_epoch = self.data_augmentation.augment(X_epoch, Y_epoch)

        for start in range(0, number_of_samples, self.batch_size):
            end = min(start + self.batch_size, number_of_samples)
            batch_indices = indices[start:end]

            x_batch = X_epoch[batch_indices]
            y_batch = Y_epoch[batch_indices]

            yield x_batch, y_batch

    def __len__(self):
        number_of_samples = self.X.shape[0]
        return (number_of_samples + self.batch_size - 1) // self.batch_size
