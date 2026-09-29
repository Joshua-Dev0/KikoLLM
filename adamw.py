import os
import json
import polars as pl
import cupy as cp
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(SCRIPT_DIR, "config.json")
with open(config_path, "r", encoding="utf-8") as file:
  config = json.load(file)
  
alpha = config["config"]["training"]["learning-rate"]
batch_size = config["config"]["training"]["batch-size"]
epoch = config["config"]["training"]["epochs"]
gradient_accumulation_steps = config["config"]["training"]["gradient-accumulation-steps"]
weight_decay = config["config"]["training"]["weight-decay"]

def gradient_descent():
  return 0