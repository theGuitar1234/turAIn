from main.turain.backend.cpu import CPU
from main.turain.neural_network.losses.binary_cross_entropy import BinaryCrossEntropyLoss

backend = CPU()
xp = backend.xp

prediction = xp.asarray([[0.8], [0.2], [0.6]], dtype=xp.float32)
true_label = xp.asarray([[1.0], [0.0], [1.0]], dtype=xp.float32)

loss_function = BinaryCrossEntropyLoss(backend)

loss = loss_function.forward_propagation(prediction, true_label)
gradient = loss_function.backward_propagation()

print("loss:", loss)
print("gradient shape:", gradient.shape)


### BCE

from main.turain.backend.cpu import CPU
from main.turain.neural_network.losses.binary_cross_entropy import BinaryCrossEntropyLoss

backend = CPU()
xp = backend.xp

prediction = xp.asarray([[0.8], [0.2], [0.6]], dtype=xp.float32)
true_label = xp.asarray([[1.0], [0.0], [1.0]], dtype=xp.float32)

loss_function = BinaryCrossEntropyLoss(backend)

loss = loss_function.forward_propagation(prediction, true_label)
gradient = loss_function.backward_propagation()

print("loss:", loss)
print("gradient shape:", gradient.shape)
