import cupy as cp

def rmsnorm(matrix):
  return 0

def transformer(block, embedded_tokens):
  token_norm = rmsnorm(embedded_tokens)
  
  query = block.AttentionBlock.Wq
  
  return tensor