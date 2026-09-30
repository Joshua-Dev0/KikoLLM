import os
import json
import cupy as cp

from bpe_tokenizer import BPE_Tokenizer 
from functions import RMSNorm, softmax, head_split
from attention import SelfAttention
from embedding import Embedding, RoPE_Embedding
from feedforward import Feedforward

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(SCRIPT_DIR, "config.json")
tokenizer_path = os.path.join(SCRIPT_DIR, "models/tokenizer.json")

with open(config_path, "r", encoding="utf-8") as file:
  config = json.load(file)

repeat = config["config"]["forward_propagation"]["repeat"]
vocab_size = config["config"]["vocabulary"]["size"]
embedding_size = config["config"]["input_layer"]["col"]
input_layer_col = config["config"]["input_layer"]["col"]
theta = config["config"]["transformer"]["position_encoding"]["theta"]
n_heads = config["config"]["transformer"]["attention"]["n-heads"]
head_dim = config["config"]["transformer"]["attention"]["head-dim"]
ffn_hidden_dim = config["config"]["transformer"]["feed_forward"]["hidden_size"]

tokenizer = BPE_Tokenizer()
embedding = Embedding()
RoPE = RoPE_Embedding()
attention = SelfAttention()
feedforward = Feedforward()
rmsnorm = RMSNorm()

def transformer(embedded_tokens):
  # Normalize embedded token
  token_norm = rmsnorm.RMSnorm(embedded_tokens)
  
  # Calculate for query, key, and value
  query = attention.query(token_norm)
  key = attention.key(token_norm)
  value = attention.value(token_norm)
  
  # Split query, key, and values to 8 heads each
  query_tensor = head_split(n_heads, head_dim, query)
  key_tensor = head_split(n_heads, head_dim, key)
  value_tensor = head_split(n_heads, head_dim, value)
  
  # Apply rotation
  query_tensor = RoPE.RoPE(query_tensor)
  key_tensor = RoPE.RoPE(key_tensor)
  
  # get attention score
  tensor_score = attention.attention_score(query_tensor, key_tensor)
  
  # scale entire tensor by 1/sqrt of head_dim
  tensor_score *= 1 / cp.sqrt(head_dim)
  
  # apply mask
  tensor_score = attention.mask(tensor_score)
  
  # apply softmax
  softmaxed_tensor = softmax(tensor_score)
  
  # dot product to value_tensor, tensor has the same size as value_tensor
  tensor = softmaxed_tensor @ value_tensor
  
  # concatenate 8 heads from (8, seq_length, 32) to (seq_length, 256)
  tensor = tensor.transpose(1, 0, 2)
  tensor = tensor.reshape(tensor.shape[0], n_heads * head_dim)
  
  # dot product with Wo from attention
  tensor = tensor @ attention.Wo
  
  # Add and save residual
  tensor = tensor + embedded_tokens
  residual = tensor

  # RMSnorm
  tensor = rmsnorm.RMSnorm(tensor)
  
  # Feedforward SwiGLU
  tensor = feedforward.SwiGLU_FFN(tensor)
  
  # add back the residual
  tensor += residual
  
  return tensor