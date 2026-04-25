from main.turain.backend.cpu import CPU
from main.turain.neural_network.losses.mean_squared_error import MeanSquaredErrorLoss

backend = CPU()
xp = backend.xp

prediction = xp.asarray([[0.3, 0.7], [0.9, 0.1]], dtype=xp.float32)
true_label = xp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=xp.float32)

loss_function = MeanSquaredErrorLoss(backend)

loss = loss_function.forward_propagation(prediction, true_label)
gradient = loss_function.backward_propagation()

print("loss:", loss)
print("gradient shape:", gradient.shape)