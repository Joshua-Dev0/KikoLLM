import os
import polars as pl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Byte Pair Encoding training is a CPU task

# Setting
vocab_size = 8000
vocab = {}
merge_rules = {}

# Placeholder
text = "My very first large language model"
space_char = "_"  # Might change depending on training example

class Tokenizer:
  def encode(self, text):
    return 0
  
  def decode(self, result):
    return 0
  
  def learn(self, data):
    dictionary = {}

    for row in data.iter_rows(named=True):
      paragraph = row["text"]

      modified_text = paragraph.replace(" ", "Ġ")
      tokens = list(modified_text)
      unique_count = len(set(tokens))
      
      print(unique_count)
      
      for token in set(tokens):
        if token not in vocab:
          vocab[token] = len(vocab)

      print("Original Vocabulary: ", vocab)
      # Count adjacent pairs
      temp = []

      # Convert tokens to pairs
      for i in range(len(tokens) - 1):
        array = [tokens[i], tokens[i + 1]]
        temp.append(array)

      # Count the number of occurences of each unqiue pairs
      for array in temp:
        key = tuple(array)

        if key in dictionary:
          dictionary[key] += 1
        else:
          dictionary[key] = 1

      # Print pairs that appear more than once
      # for key, count in dictionary.items():
      #   print(list(key), "appears", count, "times")

      # Find most frequent pair
      largest_pair = max(dictionary, key=dictionary.get)
      largest_count = dictionary[largest_pair]
      
      # Add the new pair and merge rule to vocab and merge_rules respectively
      vocab[''.join(largest_pair)] = len(vocab)
      merge_rules[largest_pair] = len(merge_rules)

      # print(
      #   "Largest pair:",
      #   list(largest_pair),
      #   "->",
      #   ''.join(largest_pair),
      #   "appears",
      #   largest_count,
      #   "times"
      # )

      # Merge the largest pair
      i = 0
      while i < len(tokens) - 1:
        if (
          tokens[i] == largest_pair[0]
          and tokens[i + 1] == largest_pair[1]
        ):
          tokens[i:i + 2] = [''.join(largest_pair)]

        i += 1

      # print(tokens)
      print("Modified Vocabulary: ", vocab)
      print("Merge Rules: ", merge_rules)
      
    return 0

data_path = os.path.join(SCRIPT_DIR, "dataset-tinystories-gpt4-clean/tinystories_gpt4_clean.parquet")
data = pl.read_parquet(data_path)
print(data.row(0))

tokenizer = Tokenizer()
tokenizer.learn(data.head(1))
