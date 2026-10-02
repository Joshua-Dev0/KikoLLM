import cupy as cp

class Feedforward:
  def initialize(self, d_model, ffn_hidden_dim):
    self.Wg = cp.random.normal(0, 0.02, (ffn_hidden_dim, d_model))
    self.Wu = cp.random.normal(0, 0.02, (ffn_hidden_dim, d_model))
    self.Wd = cp.random.normal(0, 0.02, (d_model, ffn_hidden_dim))
    
  def load(self, Wg, Wu, Wd):
    self.Wg = Wg
    self.Wu = Wu
    self.Wd = Wd
    
  def SiLU(x):
    return x * cp.reciprocal(1.0 + cp.exp(-x))
  
  def swiGLU(x):
    xWg = self.Wg @ x
    xWu = self.Wu @ x
    gate = xWg * (1 / (1 + cp.exp(-xWg)))

    return gate * xWu
  
  def swiglu_deriv():
    return 0

  def SwiGLU_FFN(x):
    xWg = x @ self.Wg
    xWu = x @ self.Wu
    hidden = self.SiLU(xWg) * xWu
    
    return hidden @ self.Wd