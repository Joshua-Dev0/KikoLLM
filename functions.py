import cupy as cp

# Config
parallel_mlps = 5

X_size = 784
hidden_count = 2
hidden_size = 64
output_size = 10


class Layer:
  def __init__(self, parallel_mlps, input_size, output_size):

    # Linear
    self.weights = cp.random.uniform(
      -0.1, 0.1,
      size=(parallel_mlps, output_size, input_size)
    )

    self.bias = cp.random.uniform(
      -0.1, 0.1,
      size=(parallel_mlps, output_size, 1)
    )

    # RMSNorm
    self.RMSnormW = cp.ones(
      (parallel_mlps, input_size, 1)
    )

    # SwiGLU
    self.Wg = cp.random.uniform(
      -0.1, 0.1,
      size=(parallel_mlps, output_size, input_size)
    )

    self.Wu = cp.random.uniform(
      -0.1, 0.1,
      size=(parallel_mlps, output_size, input_size)
    )

def initparams():
  layers = []

  for i in range(hidden_count + 1):
    if i == 0:
      layers.append(Layer(parallel_mlps, X_size, hidden_size))
    elif i < hidden_count:
      layers.append(Layer(parallel_mlps, hidden_size, hidden_size))
    else:
      layers.append(Layer(parallel_mlps, hidden_size, output_size))

  return layers


def RMSnorm(x, W):
  rms = cp.sqrt(
    cp.mean(x ** 2, axis=1, keepdims=True) + 1e-6
  )

  return (x / rms) * W


def swiGLU(x, Wg, Wu):
  xWg = Wg @ x
  xWu = Wu @ x
  gate = xWg * (1 / (1 + cp.exp(-xWg)))

  return gate * xWu

def swiglu_deriv():
  return 0

def softmax():
  return 0