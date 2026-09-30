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

def head_split(head_dim, n_heads, x):
  
  
  return 0

def transformer(embedded_tokens):
  token_norm = RMSNorm(embedded_tokens)
  
  # Split the 2D array into 8 heads each with 32 columns
  tensor_3d = matrix_2d.reshape(token_norm.shape[0], n_heads, head_dim)
  multi_head_matrix = tensor_3d.transpose(1, 0, 2)
  
  query = attention.query(multi_head_matrix)
  key = attention.key(multi_head_matrix)
  value = attention.value(multi_head_matrix)
  
  q_rotate = RoPE.RoPE(query)
  k_rotate = RoPE.RoPE(key)
  
  att_score = q_rotate @ k_rotate.T
  
  return 0