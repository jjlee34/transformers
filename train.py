import os
import time
import urllib.request

import torch

from tokenizer import BPETokenizer
from model import GPT

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "input.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH) as f:
    text = f.read()
print(f"Corpus length: {len}")
