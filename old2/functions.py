import os
import json
import cupy as cp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(SCRIPT_DIR, "config.json")

with open(config_path, "r", encoding="utf-8") as file:
  config = json.load(file)
  
class RMSNorm:
  def initialize(self, embedding_size):
    self.gamma = cp.ones(embedding_size, dtype=cp.float32)
  
  def RMSnorm(x):  
    rms = cp.sqrt(cp.mean(x ** 2, axis=1, keepdims=True) + 1e-6)
    return (x / rms) * self.gamma

def head_split(n_heads, head_dim, x):
  tensor_3d = matrix_2d.reshape(x.shape[0], n_heads, head_dim)
  multi_head_matrix = tensor_3d.transpose(1, 0, 2)
  
  return multi_head_matrix

def softmax(scores):    # Accepts a 2D matrix and softmax the rows only
  row_maxes = cp.max(scores, axis=-1, keepdims=True)
  exp_scores = cp.exp(scores - row_maxes)
  row_sums = cp.sum(exp_scores, axis=-1, keepdims=True)

  return exp_scores / row_sums