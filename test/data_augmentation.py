from turain.backend.cpu import CPU
from turain.data.batch_loader import BatchLoader
from turain.utilities import DataAugmentation
from turain.utilities import DataAugmentationType

backend = CPU()
xp = backend.xp

X = xp.arange(40, dtype=xp.float32).reshape(10, 4)
Y = xp.arange(10, dtype=xp.int64).reshape(10, 1)

augmentor = DataAugmentation(backend, DataAugmentationType.JITTER_NOISE)

train_loader = BatchLoader(
    X,
    Y,
    batch_size=3,
    backend=backend,
    shuffle=True,
    augmentor=augmentor,
)

for x_batch, y_batch in train_loader:
    print("x:", x_batch.shape, "y:", y_batch.shape)
