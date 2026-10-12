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

def get_batch(path, start, max_seq_len, batch_size, rows_per_read=64):
  seq_len = max_seq_len + 1
  total_elements = seq_len * batch_size
  token_chunks = []
  total = 0
 
  while total < total_elements:
    data = (
      pl.scan_parquet(path)
      .slice(start, rows_per_read)
      .select("tokens")
      .collect()
    )
 
    if data.is_empty():
      break
 
    for row in data["tokens"]:
      single_row = row.to_numpy()
      token_chunks.append(single_row)
 
      total += len(single_row)
      start += 1                               # only rows actually used move `start` forward
 
      if total >= total_elements:
        break
 
  if total < total_elements:
    raise ValueError("Not enough tokens remaining in the dataset.")
 
  tokens_batch = np.concatenate(token_chunks)
  batch = tokens_batch[:total_elements].reshape(batch_size, seq_len)
 
  inputs = cp.asarray(batch[:, :-1])
  targets = cp.asarray(batch[:, 1:])
 
  return inputs, targets, start

# inputs, targets, start = get_batch(tokenized_data_path, 1, 5, 1)

# print("Inputs: ", inputs)
# print("Targets: ", targets)
  
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