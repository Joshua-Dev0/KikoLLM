import os
from tokenize import BPE_Tokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(SCRIPT_DIR, "models/tokenizer.json")

tokenizer = BPE_Tokenizer()
tokenizer.load(tokenizer_path)

print(tokenizer.decode(tokenizer.encode("hello world")))