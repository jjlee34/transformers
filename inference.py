import argparse
import glob
import os

import torch

from model import GPT
from tokenizer import BPETokenizer

# generates text from a trained checkpoint without retraining
# usage:
#   python inference.py
#   python inference.py --prompt "Lily had a red ball" --num_samples 3 --temperature 0.7
#   python inference.py --ckpt checkpoints/rmsnorm.pt

def latest_checkpoint():
    # newest file in checkpoints/, falling back to the older single-file location
    ckpts = glob.glob(os.path.join("checkpoints", "*.pt"))
    return max(ckpts, key=os.path.getmtime) if ckpts else "mini_gpt.pt"

def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    if "config" in ckpt:
        config, state = ckpt["config"], ckpt["model"]
        arch = ", ".join(f"{k}={config[k]}" for k in ("norm", "pos", "mlp") if k in config)
        print(f"Loaded {path} (step {ckpt['step']}, val loss {ckpt['val_loss']:.3f}"
              f"{', ' + arch if arch else ''})")
    else:
        # older checkpoints hold only the weights; recover the sizes from their shapes.
        # n_heads can't be read from the weights, so it falls back to train.py's value
        state = ckpt
        config = dict(
            vocab_size=state["token_emb.weight"].shape[0],
            d_model=state["token_emb.weight"].shape[1],
            n_heads=8,
            n_layers=len({k.split(".")[1] for k in state if k.startswith("blocks.")}),
            max_seq_len=state["pos_emb.weight"].shape[0],
        )
        print(f"Loaded {path} (weights only; assuming n_heads=8)")
    # dropout only matters in training
    config = {**config, "dropout": 0.0}
    model = GPT(**config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None, help="set for repeatable output")
    parser.add_argument("--ckpt", default=None, help="default: newest file in checkpoints/")
    parser.add_argument("--tokenizer", default=os.path.join("data", "tokenizer.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tok = BPETokenizer()
    tok.load(args.tokenizer)
    eot_id = tok.special_tokens["<|endoftext|>"]

    model = load_model(args.ckpt or latest_checkpoint(), device)
    if model.token_emb.weight.shape[0] != tok.vocab_size:
        raise SystemExit(f"Checkpoint vocab ({model.token_emb.weight.shape[0]}) doesn't match "
                         f"tokenizer vocab ({tok.vocab_size}); wrong tokenizer for this model?")

    prompt_ids = tok.encode(args.prompt)
    for i in range(args.num_samples):
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k, eot_id=eot_id)
        print(f"\n--- sample {i + 1} ---")
        print(tok.decode([t for t in out[0].tolist() if t != eot_id]))
