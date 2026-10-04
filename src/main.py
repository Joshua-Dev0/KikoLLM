import os
import json
import cupy as cp
import numpy as np
import polars as pl

from tokenize import BPE_Tokenizer
from transformer import transformer, rmsnorm, softmax
from saveload import save, load, hash_model, tokenized_data_read, batch_read
from learning import backpropagation, adamw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(SCRIPT_DIR, "config_rev.json")
tokenizer_path = os.path.join(SCRIPT_DIR, "models/tokenizer.json")
model_path = os.path.join(SCRIPT_DIR, "models/kikollm_17M_f32.pkl")
data_path = os.path.join(SCRIPT_DIR, "dataset/tinystories_gpt4_clean.parquet")
tokenized_data = os.path.join(SCRIPT_DIR, "dataset/tinystories_tokenized.parquet")


with open(config_path, "r", encoding="utf-8") as file:
  config = json.load(file)



# Transformer specs
n_layers = config["config"]["transformer"]["n-layers"]
n_heads = config["config"]["transformer"]["attention"]["n-heads"]
head_dim = config["config"]["transformer"]["attention"]["head-dim"]
epsilon = config["config"]["transformer"]["normalization"]["epsilon"]
theta = config["config"]["transformer"]["position_encoding"]["theta"]
max_seq_len = config["config"]["context"]["max-sequence-length"]
d_model =  config["config"]["input_layer"]["col"]
vocab_size = config["config"]["input_layer"]["row"]
ffn_hidden_dim = config["config"]["transformer"]["feed_forward"]["hidden_size"]

# Training
alpha = config["config"]["training"]["learning-rate"]
lr_min = config["config"]["training"]["lr-min"]
lr_warmup_steps = config["config"]["training"]["lr-warmup-steps"]
weight_decay = config["config"]["training"]["weight-decay"]
beta1 = config["config"]["training"]["beta1"]
beta2 = config["config"]["training"]["beta2"]
grad_clip = config["config"]["training"]["grad-clip"]
batch_size = config["config"]["training"]["batch-size"]
gradient_accum_steps = config["config"]["training"]["gradient-accumulation-steps"]
epochs = config["config"]["training"]["epochs"]
precision = config["config"]["model"]["training_precision"]

# contains all the variables
class RMSnorm:
  def __init__(self, d_model):
    self.gamma = cp.ones(d_model, dtype=cp.float32)
  
  def load(self, gamma):
    self.gamma = gamma
class TransformerBlock:    # 8 transformer block
  class AttentionBlock:    # 8 heads
    def __init__(self, n_heads, head_dim, d_model):
      self.n_heads = n_heads
      self.head_dim = head_dim
      self.Wq = cp.random.normal(0.0, 0.02, size=(d_model, d_model), dtype=cp.float32)
      self.Wk = cp.random.normal(0.0, 0.02, size=(d_model, d_model), dtype=cp.float32) 
      self.Wv = cp.random.normal(0.0, 0.02, size=(d_model, d_model), dtype=cp.float32)
      self.Wo = cp.random.normal(0.0, 0.02, size=(d_model, d_model), dtype=cp.float32)
    
    def load(self, Wq, Wk, Wv, Wo):
      self.Wq = Wq
      self.Wk = Wk
      self.Wv = Wv
      self.Wo = Wo
      
  class SwiGLU:
    def __init__(self, d_model, ffn_hidden_dim):
      self.Wg = cp.random.normal(0, 0.02, (ffn_hidden_dim, d_model), dtype=cp.float32)
      self.Wu = cp.random.normal(0, 0.02, (ffn_hidden_dim, d_model), dtype=cp.float32)
      self.Wd = cp.random.normal(0, 0.02, (d_model, ffn_hidden_dim), dtype=cp.float32)
    
    def load(self, Wg, Wu, Wd):
      self.Wg = Wg
      self.Wu = Wu
      self.Wd = Wd


  def __init__(self, n_heads, head_dim, d_model, ffn_hidden_dim, theta):
    self.attention = self.AttentionBlock(n_heads, head_dim, d_model)
    self.swiglu = self.SwiGLU(d_model, ffn_hidden_dim)
    self.attention_norm = RMSnorm(d_model)
    self.ffn_norm = RMSnorm(d_model)
    self.theta = theta
    
  def load(self, Wq, Wk, Wv, Wo, Wg, Wu, Wd, attention_gamma, ffn_gamma):
    self.attention.load(Wq, Wk, Wv, Wo)
    self.swiglu.load(Wg, Wu, Wd)
    self.attention_norm.load(attention_gamma)
    self.ffn_norm.load(ffn_gamma)
  
class InferenceBlock:
  def __init__(self):
    self.transformer_blocks = []
    self.rmsnorm = RMSnorm(d_model)
    self.token_embedding = cp.random.normal(0.0, 0.02, size=(vocab_size, d_model), dtype=cp.float32)
  
  def InitializeModel(
    self,
    n_layers,
    n_heads, 
    head_dim, 
    d_model, 
    ffn_hidden_dim,
    vocab_size,
    theta
  ):
    for _ in range(n_layers):
      block = TransformerBlock(
        n_heads, 
        head_dim,
        d_model, 
        ffn_hidden_dim,
        theta
      )
      self.transformer_blocks.append(block)
      
    self.rmsnorm = RMSnorm(d_model)
    self.token_embedding = cp.random.normal(0.0, 0.02, size=(vocab_size, d_model), dtype=cp.float32)
  
  def load(self, n_layers, token_embedding, Wq, Wk, Wv, Wo, Wg,  Wu, Wd, attention_gamma, ffn_gamma, final_gamma):
    for i in range(n_layers):
      self.transformer_blocks[i].load(
        Wq[i], Wk[i], Wv[i],
        Wo[i], Wg[i], Wu[i], Wd[i],
        attention_gamma[i],
        ffn_gamma[i]
      )

    self.rmsnorm.load(final_gamma)
    self.token_embedding = token_embedding



def softmax(scores):    # softmax over the last axis
  row_maxes = cp.max(scores, axis=-1, keepdims=True)
  exp_scores = cp.exp(scores - row_maxes)
  return exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

tokenizer = BPE_Tokenizer()
tokenizer.load(tokenizer_path)
eos_id = tokenizer.tokenizer.token_to_id("<EOS>")
  
def inference(tokens, model):
  for _ in range(max_seq_len):
    tensor = model.token_embedding[cp.asarray(tokens)]
    
    for block in model.transformer_blocks:
      tensor = transformer(tensor, block)
    
    tensor_norm = rmsnorm(tensor, model.rmsnorm.gamma, epsilon)
    logits = tensor_norm @ model.token_embedding.T
    
    next_token_logits = logits[-1]
    probabilities = softmax(next_token_logits)
    probabilities = probabilities.astype(cp.float64)
    probabilities /= probabilities.sum()
    next_token_id = next_token_id = int(cp.random.choice(vocab_size, size=1, p=probabilities)[0])
    
    tokens.append(int(next_token_id))

    if (next_token_id == eos_id):
      break
  
  return tokenizer.decode(tokens), logits

def gradient_descent(model, tokenized_data_path, model_path):
  batch_size = 1024
  start = 0
  end = batch_size
  
  for i in range(epochs):
    for n in range(max_rows):
      batch = fetch_rows(tokenized_data_path, start, end)
      _, logits = inference(tokenized_text, model)
      
      
      start = end + 1
      end += batch_size - 1

      
  
  
  save(model, model_path)
  print("Hash: " + hash_model(model))


def main():
  model = InferenceBlock()
  # model.InitializeModel(n_layers, n_heads, head_dim, d_model, ffn_hidden_dim, vocab_size, theta)
  
  # print("Hash: " + hash_model(model))
  # save(model, model_path)
  model = load(model_path)
  print("Hash: " + hash_model(model))
  
  text = input("❯❯ ")
  
  tokenized_text = tokens = tokenizer.encode(text)
  result, _ = inference(tokenized_text, model)
  
  print("\n" + result + "\n")

main()