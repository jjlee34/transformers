import math
import os
import time
import urllib.request

import torch

from tokenizer import BPETokenizer
from model import GPT

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# corpus

DATA_PATH = "input.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH) as f:
    text = f.read()
print(f"Corpus length: {len(text):,}")

# tokenizer

VOCAB_SIZE = 512
num_merges = VOCAB_SIZE - 256

print(f"\nTraining BPE tokenizer (vocab_size={VOCAB_SIZE})... ")
tok = BPETokenizer()
# subset to learn good merges fast
train_text = text[:300_000]
tok.train(train_text, num_merges=num_merges)
tok.save("mini_tokenizer.json")
print(f"Tokenizer trained and saved. Actual vocab size: {tok.vocab_size}")

# encoding
print("\nEncoding full corpus...")
t0 = time.time()
ids = tok.encode(text)
print(f"Encoded {len(text):,} chars -> {len(ids):,} tokens "
      f"({len(text) / len(ids):.2f} chars/token) in {time.time() - t0:.1f}s")
data = torch.tensor(ids, dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# context length
BLOCK_SIZE = 128

def get_batch(split, batch_size):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (batch_size,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)

# model and training loop

model = GPT(
    vocab_size=tok.vocab_size,
    d_model=256,
    n_heads=8,
    n_layers=6,
    max_seq_len=BLOCK_SIZE,
    dropout=0.2
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")

BATCH_SIZE = 32
MAX_STEPS = 4000
EVAL_INTERVAL = 100

# learning rate: linear warmup, then cosine decay down to MIN_LR
MAX_LR = 1e-3
MIN_LR = 1e-4
WARMUP_STEPS = 100

def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

# weight decay on matrices only; biases and LayerNorm weights are left alone
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=MAX_LR, betas=(0.9, 0.95))

GRAD_CLIP = 1.0
CKPT_PATH = "mini_gpt.pt"

@torch.no_grad()
def estimate_loss(eval_iters=50):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split, BATCH_SIZE)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

print(f"\nTraining for {MAX_STEPS} steps on {device}...")
t0 = time.time()
best_val = float("inf")
best_step = None
for step in range(MAX_STEPS):
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    xb, yb = get_batch("train", BATCH_SIZE)
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss()
        elapsed = time.time() - t0
        marker = ""
        # keep only the checkpoint with the lowest val loss
        if losses["val"] < best_val:
            best_val, best_step = losses["val"], step
            torch.save(model.state_dict(), CKPT_PATH)
            marker = " *"
        print(f"  step {step:4d} | train loss {losses['train']:.3f} | "
              f"val loss {losses['val']:.3f} | lr {lr:.2e} | {elapsed:.1f}s elapsed{marker}")

print(f"\nTraining done in {time.time() - t0:.1f}s")
print(f"Best val loss {best_val:.3f} at step {best_step}, saved to {CKPT_PATH}")

# sample from the best checkpoint, not the final weights
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()
print("\n--- Sample generation (prompt: 'ROMEO:') ---")
prompt_ids = tok.encode("ROMEO:")
idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
out_ids = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
print(tok.decode(out_ids[0].tolist()))