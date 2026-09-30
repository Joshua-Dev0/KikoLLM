import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(SCRIPT_DIR, "models/tokenizer.json")

class BPE_Tokenizer:
  def __init__(self):
    self.tokenizer = Tokenizer(BPE())
    self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

  def encode(self, text):
    return self.tokenizer.encode(text).ids

  def decode(self, token_ids):
    return self.tokenizer.decode(token_ids)

  def learn(self, data, vocab_size):
    trainer = BpeTrainer(
      vocab_size=vocab_size,
      special_tokens=["<PAD>", "<BOS>", "<EOS>"],
      initial_alphabet=ByteLevel.alphabet()
    )

    def batch_iterator(batch_size=1000):
      for i in range(0, len(data), batch_size):
        yield data["text"][i:i + batch_size].to_list()

    self.tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(data))

  def save(self, path):
    self.tokenizer.save(path)

  def load(self, path):
    self.tokenizer = Tokenizer.from_file(path)


# data = pl.read_parquet(data_path)

# tokenizer = BPE_Tokenizer()

# tokenizer.learn(data)
# tokenizer.save(tokenizer_path)

# print(f"Tokenizer saved to: {tokenizer_path}")

# text = "Absolute Cinema"

# tokenizer.load(tokenizer_path)

# encoded = tokenizer.encode(text)
# decoded = tokenizer.decode(encoded)

# print("Original:", text)
# print("Encoded:", encoded)
# print("Decoded:", decoded)