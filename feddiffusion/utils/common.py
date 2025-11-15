# utils/common.py
import os, random, yaml, torch, numpy as np

def load_cfg(path="cfg/base.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device(cfg):
    return torch.device("cuda" if torch.cuda.is_available() and cfg.get("device","cuda")=="cuda" else "cpu")
