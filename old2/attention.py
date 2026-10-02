import cupy as cp

class SelfAttention:
  def initialize(self, embedding_size):
    self.Wq = cp.random.normal(0.0, 0.02, size=(embedding_size, embedding_size))
    self.Wk = cp.random.normal(0.0, 0.02, size=(embedding_size, embedding_size))
    self.Wv = cp.random.normal(0.0, 0.02, size=(embedding_size, embedding_size))
    self.Wo = cp.random.normal(0, 0.02, (256, 256))
    
  def load(self, Wq, Wk, Wv):
    self.Wq = Wq
    self.Wk = Wk
    self.Wv = Wv
    self.Wv = Wo
    
  def query(self, token):
    return token @ self.Wq
    
  def key(self, token):
    return token @ self.Wk
    
  def value(self, token):
    return token @ self.Wv
  
  def mask(self, scores):
    mask = cp.triu(cp.ones((scores.shape[1], scores.shape[1]), dtype=cp.bool_), k=1)
    return cp.where(mask, -cp.inf, scores)
  
  def attention_score(self, query, key):    # Across all 8 heads
    return query @ key.transpose(0, 2, 1)