import cupy as cp

from functions import softmax

def RoPE(x, head_dim, theta):
  # x: (B, H, T, head_dim)

  dimension = cp.arange(head_dim // 2, dtype=cp.float32)
  frequency = 1 / (theta ** ((2 * dimension) / head_dim))

  x_even = x[:, :, :, 0::2]
  x_odd  = x[:, :, :, 1::2]

  position = cp.arange(x.shape[2], dtype=cp.float32)
  angle = position[:, None] * frequency[None, :]

  cos = cp.cos(angle)
  sin = cp.sin(angle)

  # (T, head_dim/2)
  # broadcasts over B and H

  rotate_even = (x_even * cos[None, None, :, :] - x_odd * sin[None, None, :, :])
  rotate_odd = (x_even * sin[None, None, :, :] + x_odd * cos[None, None, :, :])

  rotate_x = cp.empty_like(x)
  rotate_x[:, :, :, 0::2] = rotate_even
  rotate_x[:, :, :, 1::2] = rotate_odd

  return rotate_x

def ff_swiglu(x, Wg, Wu, Wd):
  xWg = x @ Wg.T
  xWu = x @ Wu.T

  sigmoid = cp.reciprocal(1.0 + cp.exp(-xWg))
  gate = xWg * sigmoid
  hidden = gate * xWu
  
  output = hidden @ Wd.T

  return output, x, xWg, xWu, sigmoid, gate, hidden

def rmsnorm_cache(x, gamma, eps=1e-5):
  rms = cp.sqrt(cp.mean(x ** 2, axis=-1, keepdims=True) + eps)
  result = (x / rms) * gamma
  return result, rms

def transformer_batch(embedded_tokens, block, parameters):
  token_norm, token_rms = rmsnorm_cache(embedded_tokens, block.attention_norm.gamma)
  parameters["token_norm"] = token_norm
  parameters["token_rms"] = token_rms

  # Cross product of attention weights and token_norm
  query = token_norm @ block.attention.Wq.T
  key   = token_norm @ block.attention.Wk.T
  value = token_norm @ block.attention.Wv.T
  
  # Batch, Tokens, Embedding
  B, T, D = query.shape

  # Head splitting to 8
  query_tensor = query.reshape(B, T, block.attention.n_heads, block.attention.head_dim) \
    .transpose(0, 2, 1, 3)
  key_tensor = key.reshape(B, T, block.attention.n_heads, block.attention.head_dim) \
    .transpose(0, 2, 1, 3)
  value_tensor = value.reshape(B, T, block.attention.n_heads, block.attention.head_dim) \
    .transpose(0, 2, 1, 3)
  parameters["query"] = query_tensor
  parameters["key"] = key_tensor
  parameters["value"] = value_tensor

  # Apply rotation
  query_tensor = RoPE(query_tensor, block.attention.head_dim, block.theta)
  key_tensor = RoPE(key_tensor, block.attention.head_dim, block.theta)
  parameters["q_rot"] = query_tensor
  parameters["k_rot"] = key_tensor

  # get attention score and scale tensor elements but 1/sqrt of head_dim
  tensor_score = (query_tensor @ key_tensor.transpose(0, 1, 3, 2))
  tensor_score *= (block.attention.head_dim ** -0.5)
  parameters["attention_scores"] = tensor_score

  # Apply mask
  mask = cp.triu(cp.ones((T, T), dtype=cp.bool_), k=1)
  tensor_score = cp.where(mask, -cp.inf, tensor_score)
  parameters["attention_mask"] = mask

  # Apply softmax
  softmaxed_tensor = softmax(tensor_score)
  parameters["softmaxed_tensor"] = softmaxed_tensor

  # Dot product to value tensor
  tensor = softmaxed_tensor @ value_tensor
  parameters["attention_output"] = tensor

  # Combine 8 heads to single 2D array
  tensor = tensor.transpose(0, 2, 1, 3)
  tensor = tensor.reshape(B, T, block.attention.n_heads * block.attention.head_dim)
  parameters["merged_tensor"] = tensor

  # Dot product with Wo
  tensor = tensor @ block.attention.Wo.T

  # Add and save residual
  tensor = tensor + embedded_tokens
  residual = tensor
  parameters["attention_residual"] = residual

  # Apply normalization
  tensor, ffn_rms = rmsnorm_cache(tensor, block.ffn_norm.gamma)
  parameters["ffn_norm"] = tensor
  parameters["ffn_rms"] = ffn_rms
  
  # Feedforward
  tensor, x, xWg, xWu, sigmoid, gate, hidden = ff_swiglu(tensor, block.swiglu.Wg, block.swiglu.Wu, block.swiglu.Wd)
  parameters["ffn_x"] = x
  parameters["ffn_xWg"] = xWg
  parameters["ffn_xWu"] = xWu
  parameters["ffn_sigmoid"] = sigmoid
  parameters["ffn_gate"] = gate
  parameters["ffn_hidden"] = hidden

  # Add residual
  tensor += residual

  return tensor, parameters