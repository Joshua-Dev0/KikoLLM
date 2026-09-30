import os
import cupy as cp

class Embedding:
  def initialize(self, vocab_size, embedding_size):   # Rows -> vocab size, Cols -> embed size
    self.token_embedding = cp.random.normal(0.0, 0.02, size=(vocab_size, embedding_size))
    
  def load(self, token_embeddings):
    self.token_embedding = token_embeddings

  def lookup(self, token_ids):
    return self.token_embedding[token_ids]
  
class RoPE_Embedding:
  def load(self, theta, n_heads, head_dim):
    self.theta = theta
    self.num_heads = n_heads
    self.head_dim = head_dim
  
  def RoPE(self, x):   # x: (num_heads, T, head_dim)
    dimension = cp.arange(self.head_dim // 2, dtype=cp.float32)
    frequency = 1 / (self.theta ** ((2 * dimension) / self.head_dim))

    x_even = x[:, :, 0::2]
    x_odd  = x[:, :, 1::2]

    position = cp.arange(x.shape[1], dtype=cp.float32)

    angle = position[:, None] * frequency[None, :]

    cos = cp.cos(angle)
    sin = cp.sin(angle)

    rotate_even = (x_even * cos[None, :, :]) - (x_odd * sin[None, :, :])
    rotate_odd = (x_even * sin[None, :, :]) + (x_odd * cos[None, :, :])

    rotate_x = cp.empty_like(x)
    rotate_x[:, :, 0::2] = rotate_even
    rotate_x[:, :, 1::2] = rotate_odd

    return rotate_x
