import os
import json
import cupy as cp

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# config_path = os.path.join(SCRIPT_DIR, "config.json")
# with open(config_path, "r", encoding="utf-8") as file:
#   config = json.load(file)
  
# vocab_size = config["config"]["vocabulary"]["size"]
# embedding_size = config["config"]["input_layer"]["col"]
# theta = config["config"]["transformer"]["position_encoding"]["theta"]
# n_heads = config["config"]["transformer"]["attention"]["n-heads"]
# head_dim = config["config"]["transformer"]["attention"]["head-dim"]

class Embedding:
  def initialize(self, vocab_size, embedding_size):   # Rows -> vocab size, Cols -> embed size
    self.token_embedding = cp.random.normal(0.0, 0.02, size=(vocab_size, embedding_size))
    
  def load(self, token_embeddings):
    self.token_embedding = token_embeddings

  def lookup(self, token_ids):
    return self.token_embedding[token_ids]
  
class RoPE_Embedding:
  def initialize(self, theta, n_heads, head_dim):
    self.theta = theta
    self.num_heads = n_heads
    self.head_dim = head_dim
  
  def RoPE(self, x):   # Each row is the token
    dimension = cp.arange(self.head_dim // 2, dtype=cp.float32)
    frequency = 1 / (self.theta ** ((2 * dimension) / self.head_dim))
    
    x_even = x[:, 0::2]   # Gets only the x part of the pair
    x_odd = x[:, 1::2]    # Gets only the y part of the pair
    
    position = cp.arange(x.shape[0], dtype=cp.float32) 
    angle = position[:, None] * frequency[None, :]
    
    cos = cp.cos(angle)
    sin = cp.sin(angle)
    
    rotate_even = (x_even * cos) - (x_odd * sin)
    rotate_odd = (x_even * sin) + (x_odd * cos)
    
    rotate_x = cp.empty_like(x)
    rotate_x[:, 0::2] = rotate_even
    rotate_x[:, 1::2] = rotate_odd
    
    return rotate_x
  
# embedding = Embedding(vocab_size, embedding_size)
# embed = embedding.forward(0)

# print(embed)

# embed = Embedding()

# embed.initialize(5, 5)
# print(embed.token_embedding)
# print(embed.lookup(0))