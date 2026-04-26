from utilities import core_method
from utilities import TrainDefaults
from utilities import DataAugmentationType

class DataAugmentation:
    def __init__(self, backend, augmentation_type):
        self.augmentation_type = augmentation_type
        self.backend = backend
    
    @core_method
    def augment(self, X, Y, config=None):
        xp = self.backend.xp
        random_range = xp.random.default_range()
        
        if config is None:
            config = TrainDefaults()
        
        measurement_strength = config.measurement_strength
        jitter_strength = config.jitter_strength
        same_class_augmentation_low_alpha = config.same_class_augmentation_low_alpha
        same_class_augmentation_high_alpha = config.same_class_augmentation_high_alpha

        X = xp.asarray(X, dtype=float)
        Y = xp.asarray(Y)

        X_aug = X.copy()
        Y_aug = Y.copy()

        match self.augmentation_type:
            case DataAugmentationType.JITTER_NOISE:
                noise = random_range.uniform(-jitter_strength, jitter_strength, size=X_aug.shape)
                X_aug = X_aug + noise

            case DataAugmentationType.MEASUREMENT_NOISE:
                scale = random_range.uniform(
                    1.0 - measurement_strength,
                    1.0 + measurement_strength,
                    size=X_aug.shape
                )
                X_aug = xp.maximum(0.0, X_aug * scale)

            case DataAugmentation.SAME_CLASS_INTERPOLATION:
                if Y_aug.shape[1] == 1:
                    labels = Y_aug.flatten().astype(int)
                else:
                    labels = xp.argmax(Y_aug, axis=1)

                synthetic_x = []
                synthetic_y = []

                for label in xp.unique(labels):
                    indices = xp.where(labels == label)[0]
                    num_synthetic = len(indices) // 2

                    for _ in range(num_synthetic):
                        if len(indices) >= 2:
                            i1, i2 = random_range.choice(indices, size=2, replace=False)
                        else:
                            i1 = i2 = indices[0]

                        alpha = random_range.uniform(same_class_augmentation_low_alpha, same_class_augmentation_high_alpha)

                        x_new = alpha * X_aug[i1] + (1.0 - alpha) * X_aug[i2]
                        y_new = Y_aug[i1].copy()

                        synthetic_x.append(x_new)
                        synthetic_y.append(y_new)

                if synthetic_x:
                    synthetic_x = xp.stack(synthetic_x, axis=0)
                    synthetic_y = xp.stack(synthetic_y, axis=0)

                    X_aug = xp.concatenate([X_aug, synthetic_x], axis=0)
                    Y_aug = xp.concatenate([Y_aug, synthetic_y], axis=0)

            case _:
                raise ValueError(f"Unknown Data Augmentation Type, supported values are : {self.DataAugmentation.MEASUREMENT_NOISE}, {self.DataAugmentation.JITTER_NOISE}, {self.DataAugmentation.SAME_CLASS_INTERPOLATION}")

        return X_aug, Y_aug