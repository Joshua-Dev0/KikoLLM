import os
import json
import polars as pl
from tqdm import tqdm
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Settings
vocab_size = 8000
checkpoint_interval = 100
training_stories = 100000

start_text = "<BOS>"
end_text = "<EOS>"
pad_text = "<PAD>"

data_path = os.path.join(SCRIPT_DIR, "dataset-tinystories-gpt4-clean/tinystories_gpt4_clean.parquet")
tokenizer_path = os.path.join(SCRIPT_DIR, "tokenizer.json")
checkpoint_path = os.path.join(SCRIPT_DIR, "tokenizer_checkpoint.json")


class BPE_Learn:
  def pairs(self, tokens):
    pairs_array = []

    for i in range(len(tokens) - 1):
      pair = (tokens[i], tokens[i + 1])
      pairs_array.append(pair)

    return pairs_array

  def replace(self, pair, tokens):
    new_tokens = []
    i = 0

    while i < len(tokens):
      if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
        new_token = tokens[i] + tokens[i + 1]
        new_tokens.append(new_token)
        i += 2
      else:
        new_tokens.append(tokens[i])
        i += 1

    return new_tokens

  def learn(self, vocab, merge_rules, pair):
    new_token = pair[0] + pair[1]

    if new_token not in vocab:
      vocab[new_token] = len(vocab)

    if pair not in merge_rules:
      merge_rules[pair] = len(merge_rules)

    return vocab, merge_rules


class Tokenizer:
  def encode(self, text, vocab, merge_rules):
    tokens = [bytes([byte]) for byte in text.encode("utf-8")]

    sorted_rules = sorted(merge_rules.items(), key=lambda item: item[1])

    for pair, rank in sorted_rules:
      tokens = BPE_Learn().replace(pair, tokens)

    return [vocab[token] for token in tokens]

  def decode(self, token_ids, vocab):
    id_to_token = {token_id: token for token, token_id in vocab.items()}
    byte_result = bytearray()

    for token_id in token_ids:
      token = id_to_token[token_id]

      if isinstance(token, bytes):
        byte_result.extend(token)

    return byte_result.decode("utf-8")

  def save(self, vocab, merge_rules, path):
    data = {
      "vocab": {token.hex() if isinstance(token, bytes) else token: token_id for token, token_id in vocab.items()},
      "merge_rules": [{"pair": [a.hex(), b.hex()], "rank": rank} for (a, b), rank in merge_rules.items()]
    }

    with open(path, "w", encoding="utf-8") as file:
      json.dump(data, file, indent=2)

  def load(self, path):
    with open(path, "r", encoding="utf-8") as file:
      data = json.load(file)

    vocab = {}

    for token, token_id in data["vocab"].items():
      if token.startswith("<") and token.endswith(">"):
        vocab[token] = token_id
      else:
        vocab[bytes.fromhex(token)] = token_id

    merge_rules = {}

    for rule in data["merge_rules"]:
      pair = (bytes.fromhex(rule["pair"][0]), bytes.fromhex(rule["pair"][1]))
      merge_rules[pair] = rule["rank"]

    return vocab, merge_rules

  def learn(self, vocab, merge_rules, data):
    bpe = BPE_Learn()

    # Add special tokens
    if start_text not in vocab:
      vocab[start_text] = len(vocab)

    if end_text not in vocab:
      vocab[end_text] = len(vocab)

    if pad_text not in vocab:
      vocab[pad_text] = len(vocab)

    # Tokenize dataset
    tokenized_data = []

    for paragraph in tqdm(data["text"], desc="Tokenizing"):
      tokens = [bytes([byte]) for byte in paragraph.encode("utf-8")]
      tokenized_data.append(tokens)

    print(f"Initial vocabulary: {len(vocab)}")
    print(f"Stories: {len(tokenized_data)}")

    # Calculate remaining merges
    remaining_merges = vocab_size - len(vocab)

    with tqdm(total=remaining_merges, initial=len(merge_rules), desc="Training BPE") as pbar:
      while len(vocab) < vocab_size:

        # Count all pairs
        pair_counts = Counter()

        for tokens in tokenized_data:
          pair_counts.update(bpe.pairs(tokens))

        # Stop if there are no pairs
        if not pair_counts:
          print("No more pairs available.")
          break

        # Find most frequent pair
        frequent_pair = pair_counts.most_common(1)[0][0]

        # Add new token and merge rule
        old_vocab_size = len(vocab)
        vocab, merge_rules = bpe.learn(vocab, merge_rules, frequent_pair)

        # Apply merge
        for i in range(len(tokenized_data)):
          tokenized_data[i] = bpe.replace(frequent_pair, tokenized_data[i])

        # Update progress
        if len(vocab) > old_vocab_size:
          pbar.update(1)

        # Save checkpoint
        if len(merge_rules) > 0 and len(merge_rules) % checkpoint_interval == 0:
          tokenizer.save(vocab, merge_rules, checkpoint_path)
          print(f"\nCheckpoint saved at {len(merge_rules)} merges.")

    return vocab, merge_rules


# Load dataset
print("Loading dataset...")

data = pl.read_parquet(data_path)

print(f"Dataset contains {len(data)} stories.")


# Limit training data
data = data.head(training_stories)

print(f"Using {len(data)} stories for BPE training.")


# Create tokenizer
tokenizer = Tokenizer()


# Initialize byte vocabulary
vocab = {bytes([i]): i for i in range(256)}
merge_rules = {}


# Load checkpoint if available
if os.path.exists(checkpoint_path):
  print("Checkpoint found.")
  print("Loading checkpoint...")

  vocab, merge_rules = tokenizer.load(checkpoint_path)

  print(f"Loaded vocabulary: {len(vocab)}")
  print(f"Loaded merge rules: {len(merge_rules)}")
else:
  print("No checkpoint found.")
  print("Starting BPE training from scratch.")


# Train tokenizer
vocab, merge_rules = tokenizer.learn(vocab, merge_rules, data)


# Save final tokenizer
tokenizer.save(vocab, merge_rules, tokenizer_path)

print(f"Vocabulary size: {len(vocab)}")
print(f"Merge rules: {len(merge_rules)}")
print(f"Tokenizer saved to: {tokenizer_path}")


# Test tokenizer
test_text = "My very first large language model"

encoded = tokenizer.encode(test_text, vocab, merge_rules)
decoded = tokenizer.decode(encoded, vocab)

print("Test:")
print("Original:", test_text)
print("Encoded:", encoded)
print("Decoded:", decoded)