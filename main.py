import os
import json
import cupy as cp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(SCRIPT_DIR, "config.json")
with open(config_path, "r", encoding="utf-8") as file:
  config = json.load(file)

def inference():
  return 0



def main():
  return 0

main()