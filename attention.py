import os
import json
import cupy as cp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(SCRIPT_DIR, "config.json")

with open(config_path, "r", encoding="utf-8") as file:
  config = json.load(file)
  
embedding_size = config["config"]["input_layer"]["col"]
class SelfAttention:
  def __init__(self):
    self.embedding_size = embedding_size        # Row & Column

    self.Wq = cp.random.normal(0.0, 0.02, size=(embedding_size, embedding_size))
    self.Wk = cp.random.normal(0.0, 0.02, size=(embedding_size, embedding_size))
    self.Wv = cp.random.normal(0.0, 0.02, size=(embedding_size, embedding_size))
    
  def query(self, token):
    return token @ self.Wq
    
  def key(self, token):
    return token @ self.Wk
    
  def value(self, token):
    return token @ self.Wv
  
self_attention = SelfAttention()
print(self_attention.Wq)