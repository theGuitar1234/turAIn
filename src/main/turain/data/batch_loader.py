from utilities import core_method


class BatchLoader:
    def __init__(self, x, y, batch_size, backend, shuffle=True, data_augmentation=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.backend = backend
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation

    @core_method
    def __iter__(self):
        xp = self.backend
        number_of_samples = self.X.shape[0]

        indices = xp.arange(number_of_samples)

        if self.shuffle:
            xp.random.shuffle(indices)

        for start in range(0, number_of_samples, self.batch_size):
            end = min(start + self.batch_size, number_of_samples)
            batch_indices = indices[start:end]

            x_batch = self.X[batch_indices]
            y_batch = self.Y[batch_indices]

            if self.data_augmentation:
                x_batch, y_batch = self.data_augmentation.augment(x_batch, y_batch)

            yield x_batch, y_batch


if __name__ == "__main__":
    pass
