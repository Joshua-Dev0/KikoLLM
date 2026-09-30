import cupy as cp

class SelfAttention:
  def initialize(self, embedding_size):
    self.Wq = cp.random.normal(0.0, 0.02, size=(embedding_size, embedding_size))
    self.Wk = cp.random.normal(0.0, 0.02, size=(embedding_size, embedding_size))
    self.Wv = cp.random.normal(0.0, 0.02, size=(embedding_size, embedding_size))
    
  def load(self, Wq, Wk, Wv):
    self.Wq = Wq
    self.Wk = Wk
    self.Wv = Wv
    
  def query(self, token):
    return token @ self.Wq
    
  def key(self, token):
    return token @ self.Wk
    
  def value(self, token):
    return token @ self.Wv
  
  def mask(self, scores):
    mask = cp.triu(cp.ones(scores.shape), k=1)
    masked_scores = cp.where(mask == 1, -cp.inf, scores)
    return masked_scores
  
  def attention_score(self, query, key):
    return query @ key.T
  
  
# def attention(embedded_token):
#   self_attention = SelfAttention()
  
