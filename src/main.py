import os
import json
import cupy as cp

from bpe_tokenizer import BPE_Tokenizer 
from functions import RMSNorm, softmax
from attention import SelfAttention
from embedding import Embedding, RoPE_Embedding
from feed_forward import FeedForwardNetwork
from transformer import transformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(SCRIPT_DIR, "config.json")
tokenizer_path = os.path.join(SCRIPT_DIR, "models/tokenizer.json")

with open(config_path, "r", encoding="utf-8") as file:
  config = json.load(file)
  
repeat = config["config"]["forward_propagation"]["repeat"]
vocab_size = config["config"]["vocabulary"]["size"]
embedding_size = config["config"]["input_layer"]["col"]
theta = config["config"]["transformer"]["position_encoding"]["theta"]
n_heads = config["config"]["transformer"]["attention"]["n-heads"]
head_dim = config["config"]["transformer"]["attention"]["head-dim"]
max_seq_len = config["config"]["context"]["max-sequence-length"]

tokenizer = BPE_Tokenizer()
tokenizer.load(tokenizer_path)
eos_id = tokenizer.tokenizer.token_to_id("<EOS>")

embedding = Embedding()
rmsnorm = RMSNorm()

def initialize():
  return 0

def load(model_path):
  return 0

def inference(text):
  tokens = tokenizer.encode(text)
  original_length = len(tokens)
  max_new_tokens = max_seq_len - len(tokens)
  
  for _ in range(max_new_tokens):
    tensor = embedding.lookup(tokens)
    
    for i in range(repeat):
      tensor = transformer(tensor)
    
    tensor_norm = rmsnorm.RMSnorm(tensor)
    logits = tensor_norm @ embedding.token_embedding.T
    
    next_token_logits = logits[-1]
    probabilities = softmax(next_token_logits)
    next_token_id = cp.argmax(cp.cumsum(probabilities) > cp.random.rand())
    
    tokens.append(int(next_token_id))

    if (next_token_id == eos_id):
      break
  
  generated_tokens = tokens[original_length:]
  
  return tokenizer.decode(generated_tokens)

def main():
  text = input("❯❯ ")
  
  result = inference(text)
  
  output = result + " ❮❮"
  terminal_width = os.get_terminal_size().columns
  print(output.rjust(terminal_width))
  
initialize()
main()

