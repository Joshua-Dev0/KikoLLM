import os
import json
import cupy as cp

from bpe_tokenizer import BPE_Tokenizer 
from functions import RMSNorm, softmax
from attention import SelfAttention
from embedding import Embedding, RoPE_Embedding
from feed_forward import FeedForwardNetwork

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

tokenizer = BPE_Tokenizer()
embedding = Embedding()
RoPE = RoPE_Embedding()
attention = SelfAttention()

def initialize():
  embedding.initialize(vocab_size, embedding_size)
  RoPE.initialize(theta, n_heads, head_dim)
  attention.initialize(embedding_size)
  
def load():
  return 0
  
# def transformer(embedded_tokens):
#   token_norm = RMSNorm(embedded_tokens)
  
#   query = attention.query(token_norm)
#   key = attention.key(token_norm)
#   value = attention.value(token_norm)
  
  
  
#   q_rotate = RoPE.RoPE(query)
#   k_rotate = RoPE.RoPE(key)
  
#   att_score = q_rotate @ k_rotate.T
  
#   return 0

def inference(text, repeat):
  # Step 1: Tokenize raw string to integers
  tokenizer.load(tokenizer_path)
  tokens = tokenizer.encode(text)
  print(tokens)
  
  # Step 2: Token embedding
  embedded_tokens = embedding.lookup(tokens)
  
  # Step 3: Feed to transformer
  output = transformer(embedded_tokens)
  
  return "Sup"


def main():
  text = input("❯❯ ")
  
  result = inference(text, repeat)
  
  output = result + " ❮❮"
  terminal_width = os.get_terminal_size().columns
  print(output.rjust(terminal_width))
  
  return 0

initialize()
main()