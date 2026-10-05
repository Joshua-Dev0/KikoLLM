import cupy as cp

def rmsnorm(x, gamma, eps=1e-5):
  rms = cp.sqrt(cp.mean(x ** 2, axis=-1, keepdims=True) + eps)
  return (x / rms) * gamma

def ff_swiglu(x, Wg, Wu, Wd):
  xWg = x @ Wg.T
  xWu = x @ Wu.T

  gate = xWg * cp.reciprocal(1.0 + cp.exp(-xWg))
  hidden = gate * xWu

  return hidden @ Wd.T

def softmax(scores):    # softmax over the last axis
  row_maxes = cp.max(scores, axis=-1, keepdims=True)
  exp_scores = cp.exp(scores - row_maxes)
  return exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)