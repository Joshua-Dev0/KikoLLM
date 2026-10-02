import cupy as cp

def rmsnorm(x, gamma, eps=1e-5):
  rms = cp.sqrt(cp.mean(x ** 2, axis=-1, keepdims=True) + eps)
  return (x / rms) * gamma

def RoPE(x, head_dim, theta):   # x: (num_heads, T, head_dim)
  dimension = cp.arange(head_dim // 2, dtype=cp.float32)
  frequency = 1 / (theta ** ((2 * dimension) / head_dim))

  x_even = x[:, :, 0::2]
  x_odd  = x[:, :, 1::2]

  position = cp.arange(x.shape[1], dtype=cp.float32)

  angle = position[:, None] * frequency[None, :]

  cos = cp.cos(angle)
  sin = cp.sin(angle)

  rotate_even = (x_even * cos[None, :, :]) - (x_odd * sin[None, :, :])
  rotate_odd = (x_even * sin[None, :, :]) + (x_odd * cos[None, :, :])

  rotate_x = cp.empty_like(x)
  rotate_x[:, :, 0::2] = rotate_even
  rotate_x[:, :, 1::2] = rotate_odd

  return rotate_x

def ff_swiglu(x, Wg, Wu, Wd):
  xWg = x @ Wg.T
  xWu = x @ Wu.T

  gate = xWg * cp.reciprocal(1.0 + cp.exp(-xWg))
  hidden = gate * xWu

  return hidden @ Wd.T

def softmax(scores):    # softmax over the last axis
  row_maxes = cp.max(scores, axis=-1, keepdims=True)
  exp_scores = cp.exp(scores - row_maxes)
  return exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

def transformer(embedded_tokens, block):
  token_norm = rmsnorm(embedded_tokens, block.attention_norm.gamma)
  
  # Cross product of attention weights and token_norm
  query = token_norm @ block.attention.Wq.T
  key = token_norm @ block.attention.Wk.T
  value = token_norm @ block.attention.Wv.T
  
  # Head splitting to 8
  query_tensor = query \
    .reshape(embedded_tokens.shape[0], block.attention.n_heads, block.attention.head_dim) \
    .transpose(1, 0, 2)
    
  key_tensor = key \
    .reshape(embedded_tokens.shape[0], block.attention.n_heads, block.attention.head_dim) \
    .transpose(1, 0, 2)
    
  value_tensor = value \
    .reshape(embedded_tokens.shape[0], block.attention.n_heads, block.attention.head_dim) \
    .transpose(1, 0, 2)
    
  # Apply rotation
  query_tensor = RoPE(query_tensor, block.attention.head_dim, block.theta)
  key_tensor = RoPE(key_tensor, block.attention.head_dim, block.theta)
  
  # get attention score
  tensor_score = query_tensor @ key_tensor.transpose(0, 2, 1)
  
  # Scale tensor elements but 1/sqrt of head_dim
  tensor_score *= block.attention.head_dim ** -0.5
  
  # Apply mask
  mask = cp.triu(cp.ones((tensor_score.shape[1], tensor_score.shape[1]), dtype=cp.bool_), k=1)
  tensor_score_masked = cp.where(mask, -cp.inf, tensor_score)
  
  # Apply softmax
  softmaxed_tensor = softmax(tensor_score_masked)
  
  # Dot product to value tensor
  tensor = softmaxed_tensor @ value_tensor
  
  # Combine 8 heads to single 2D array
  tensor = tensor.transpose(1, 0, 2)
  tensor = tensor.reshape(tensor.shape[0], block.attention.n_heads * block.attention.head_dim)
  
  # Dot product with Wo
  tensor = tensor @ block.attention.Wo.T
  
  # Add and save residual
  tensor = tensor + embedded_tokens
  residual = tensor
  
  # Apply normalization
  tensor = rmsnorm(tensor, block.ffn_norm.gamma)
  
  # Feedforward
  tensor = ff_swiglu(tensor, block.swiglu.Wg, block.swiglu.Wu, block.swiglu.Wd)
  
  # Add residual
  tensor += residual
  
  return tensor