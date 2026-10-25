import math
import os
import time
import numpy as np
import torch

from tokenizer import BPETokenizer
from model import GPT

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# data: token ids written by prepare_data.py

DATA_DIR = "data"
if not os.path.exists(os.path.join(DATA_DIR, "train.bin")):
    raise SystemExit("No prepared data found. Run `python prepare_data.py` first.")

tok = BPETokenizer()
tok.load(os.path.join(DATA_DIR, "tokenizer.json"))
print(f"Tokenizer vocab size: {tok.vocab_size}")

# memmap reads windows straight from disk, so the dataset doesn't have to fit in RAM
train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
print(f"Train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

# context length; most TinyStories fit in 256 tokens
BLOCK_SIZE = 256

def get_batch(split, batch_size):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(d[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + BLOCK_SIZE + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# model and training loop

model = GPT(
    vocab_size=tok.vocab_size,
    d_model=256,
    n_heads=8,
    n_layers=6,
    max_seq_len=BLOCK_SIZE,
    # each token is seen less than once, so there is nothing to memorize and no need for dropout
    dropout=0.0
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
print("\n--- Sample generation (prompt: 'Once upon a time') ---")
prompt_ids = tok.encode("Once upon a time")
idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
out_ids = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
print(tok.decode(out_ids[0].tolist()))