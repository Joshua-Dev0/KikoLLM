import cupy as cp

from forwardpass import transformer_batch

def adamw(n_layers, dtoken_embedding, dWq, dWk, dWv, dWo, dWg, dWu, dWd, dattention_gamma, dffn_gamma, dfinal_gamma):


  return token_embedding, Wq, Wk, Wv, Wo, Wg, Wu, Wd, attention_gamma, ffn_gamma, final_gamma

def cross_entropy(logits, target):
  batch_size, seq_len, vocab_size = logits.shape

  logits = logits.reshape(batch_size * seq_len, vocab_size)
  target = target.reshape(batch_size * seq_len).astype(cp.int32)

  # Numerically stable log-softmax
  logits = logits - cp.max(logits, axis=1, keepdims=True)

  log_probs = logits - cp.log(cp.sum(cp.exp(logits), axis=1, keepdims=True))

  # Select the log-probability of each correct token
  correct_log_probs = log_probs[cp.arange(B * T), target]

  loss = -cp.mean(correct_log_probs)

  return loss

def backpropagation(model, loss):
  # Loss
  
  return model