import os
import cupy as cp
import polars as pl
import numpy as np
import pickle
import hashlib
from tqdm import tqdm
import math
import pyarrow as pa
import pyarrow.parquet as pq

from tokenize import BPE_Tokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(SCRIPT_DIR, "dataset/tinystories_gpt4_clean.parquet")
tokenized_data_path = os.path.join(SCRIPT_DIR, "dataset/tinystories_tokenized.parquet")
tokenizer_path = os.path.join(SCRIPT_DIR, "models/tokenizer.json")

def tokenize_parquet(input_path, output_path, tokenizer, batch_size=1000):
  df = pl.read_parquet(input_path)
  writer = None

  try:
    for start in tqdm(
      range(0, df.height, batch_size),
      desc="Tokenizing"
    ):
      batch = df.slice(start, batch_size)

      tokens = [
        tokenizer.encode(text)
        for text in batch["text"]
      ]

      tokenized_batch = pl.DataFrame({
        "tokens": tokens
      })

      table = tokenized_batch.to_arrow()

      if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema)

      writer.write_table(table)

      del batch
      del tokens
      del tokenized_batch
      del table

  finally:
    if writer is not None:
      writer.close()
  
# tokenizer = BPE_Tokenizer()
# tokenizer.load(tokenizer_path)
# tokenize_parquet(data_path, tokenized_data_path, tokenizer, 1000)
        
def tokenized_data_read(tokenizer, tokenized_path, row=0):
  data = pl.read_parquet(tokenized_path)
  tokens = data["tokens"][row].to_numpy()
  
  return tokens

# tokenizer = BPE_Tokenizer()
# tokenizer.load(tokenizer_path)
# tokens = tokenizer.decode(tokenized_data_read(tokenizer, tokenized_data_path, row=10))
# print("Token IDs:")
# print(tokens)
# text = tokenizer.decode(tokens)
# print("\nDecoded text:")
# print(text)

def get_batch(path, start, batch_size):
  data = (
    pl.scan_parquet(path)
    .slice(start, batch_size)
    .select("tokens")
    .collect()
  )

  return [
    tokens.to_numpy()
    for tokens in data["tokens"]
  ]
  
# tokenizeData(data_path, tokenized_data_path)
# print(tokenized_data_read(tokenizer_path, tokenized_data_path, row=1399))

# tokenizer = BPE_Tokenizer()
# tokenizer.load(tokenizer_path)
# batch = get_batch(tokenized_data_path, start=0, batch_size=2)

# print(len(batch))
# print("Row 0:", tokenizer.decode(batch[0].tolist()))
# print("Row 1:", tokenizer.decode(batch[1].tolist()))

def save(model, path):
  with open(path, "wb") as file:
    pickle.dump(model, file)

def load(path):
  with open(path, "rb") as file:
    return pickle.load(file)
  
def hash_model(model):
  data = pickle.dumps(model)
  return hashlib.sha256(data).hexdigest()