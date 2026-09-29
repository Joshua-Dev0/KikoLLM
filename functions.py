import os
import json
import cupy as cp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(SCRIPT_DIR, "config.json")

with open(config_path, "r", encoding="utf-8") as file:
  config = json.load(file)
  
class swiGLU:
  def __init__(self):
    return 0
  
  def swiGLU(x, Wg, Wu):
    xWg = Wg @ x
    xWu = Wu @ x
    gate = xWg * (1 / (1 + cp.exp(-xWg)))

    return gate * xWu
  
  def swiglu_deriv():
    return 0
  
class RMSNorm:
  def __init__(self):
    return 0
  
  def RMSnorm(x, W):
    rms = cp.sqrt(cp.mean(x ** 2, axis=1, keepdims=True) + 1e-6)
    return (x / rms) * W


def softmax():
  return 0