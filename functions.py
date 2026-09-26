import cupy as cp

# Config
X_size = 784
hidden_count = 2
hidden_size = 64
output_size = 10

class Layer:
  def __init__(self, input_size, output_size):
    self.weights = cp.random.uniform(-0.1, 0.1, size=(output_size, input_size))
    self.bias = cp.zeros((output_size, 1))
    
    # RMSNorm
    self.RMSnormW = cp.ones((input_size, 1))

    # SwiGLU
    self.Wg = cp.random.uniform(-0.1, 0.1, size=(output_size, input_size))
    self.Wu = cp.random.uniform(-0.1, 0.1, size=(output_size, input_size))

def initparams():
  layers = []

  for i in range(hiddenl_count + 1):
    if i == 0:
      layers.append(Layer(input_size, hiddenl_size))

    elif i < hiddenl_count:
      layers.append(Layer(hiddenl_size, hiddenl_size))

    else:
      layers.append(Layer(hiddenl_size, output_size))

  return layers

def RMSnorm(x, W):
  rms = cp.sqrt(cp.mean(x ** 2, axis=1, keepdims=True) + 1e-6)
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