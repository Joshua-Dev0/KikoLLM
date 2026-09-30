import cupy as cp

class SwiGLU:
  def __init__(self):
    return 0
  
  def swiGLU(x, Wg, Wu):
    xWg = Wg @ x
    xWu = Wu @ x
    gate = xWg * (1 / (1 + cp.exp(-xWg)))

    return gate * xWu
  
  def swiglu_deriv():
    return 0

def FeedForwardNetwork(X, layers, Wg, Wu):
  parallel_cols = X.shape[1]
  A = [X]
  Z = []

  for i in range(len(layers)):
    z = layers[i].weights @ A[i] + layers[i].bias
    a = swiGLU(z, layers[i].Wg, layers[i].Wu)

    Z.append(z)
    A.append(a)

  return A, Z


# Backprop and update Needs revision
# def backward_propagation(Y_onehot: cp.ndarray, A, Z, layers):
#   dW = [None] * len(layers)
#   dB = [None] * len(layers)

#   # Output layer
#   dZ = A[-1] - Y_onehot

#   # Traverse layers backwards
#   for i in reversed(range(len(layers))):

#     # Gradients for current layer
#     dW[i] = dZ @ A[i].T
#     dB[i] = dZ

#     # Propagate gradient to previous layer
#     if i > 0:
#       dA = layers[i].weights.T @ dZ
#       dZ = dA * relu_deriv(Z[i - 1])

#   return dW, dB

# def update_params(alpha, layers, dW, dB):
#   for i in range(len(layers)):
#     layers[i].weights -= alpha * dW[i]
#     layers[i].bias -= alpha * dB[i]

#   return layers