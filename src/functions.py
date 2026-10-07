import cupy as cp

def softmax(scores):    # softmax over the last axis
  row_maxes = cp.max(scores, axis=-1, keepdims=True)
  exp_scores = cp.exp(scores - row_maxes)
  return exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)
