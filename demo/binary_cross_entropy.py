from main.turain.neural_network.losses.binary_cross_entropy import BinaryCrossEntropyLoss
from main.turain.backend.cpu import CPU

backend = CPU()

xp = backend.xp

predictions = [[0.8], [0.2], [0.6]]
true_labels = [[1.0], [0.0], [1.0]]

prediction = xp.asarray(predictions, dtype=xp.float32)
true_label = xp.asarray(true_labels, dtype=xp.float32)

loss_function = BinaryCrossEntropyLoss(backend)

loss = loss_function.forward_propagation(prediction, true_label)
gradient = loss_function.backward_propagation()

print(f"Loss : {loss}")
print(f"Gradient Shape : {gradient.shape}")
