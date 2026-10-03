import os
import cupy as cp
import polars as pl
import numpy as np
import pickle
import hashlib
from tqdm import tqdm
import math

from tokenize import BPE_Tokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(SCRIPT_DIR, "dataset/tinystories_gpt4_clean.parquet")
tokenized_data_path = os.path.join(SCRIPT_DIR, "dataset/tinystories_tokenized.parquet")
tokenizer_path = os.path.join(SCRIPT_DIR, "models/tokenizer.json")

def tokenizeData(parquet_path, output_path):
  tokenizer = BPE_Tokenizer()
  tokenizer.load(tokenizer_path)

  dataset = (pl.scan_parquet(parquet_path).select("text"))
  total_rows = (dataset.select(pl.len()).collect().item())

  chunk_size = 1024
  total_batches = math.ceil(total_rows / chunk_size)
  os.makedirs(output_path, exist_ok=True)
  batches = dataset.collect_batches(chunk_size=chunk_size)

  for batch_number, batch in enumerate(tqdm(batches, total=total_batches, desc="Tokenizing corpus")):
    tokenized_rows = []

    for text in batch["text"]:
      tokens = tokenizer.encode(text)
      tokenized_rows.append(tokens)

    output_batch = pl.DataFrame({"tokens": tokenized_rows})
    output_batch.write_parquet(f"{output_path}/part_{batch_number:05d}.part")
        
def tokenized_data_read(tokenized_path, row=0):
  tokenizer = BPE_Tokenizer()
  tokenizer.load(tokenizer_path)

  data = pl.read_parquet(tokenized_path)
  tokens = data["tokens"][row].to_numpy()

  tokens = tokens.reshape(1, -1)

  # print("Token IDs:")
  # print(tokens)
  # text = tokenizer.decode(tokens)
  # print("\nDecoded text:")
  # print(text)
  
  return tokens
  
# tokenizeData(data_path, tokenized_data_path)
print(tokenized_data_read(tokenized_data_path, row=1399))

def save(model, path):
  with open(path, "wb") as file:
    pickle.dump(model, file)

def load(path):
  with open(path, "rb") as file:
    return pickle.load(file)
  
def hash_model(model):
  data = pickle.dumps(model)
  return hashlib.sha256(data).hexdigest()