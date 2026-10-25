import os
import time
import urllib.request

import numpy as np

from tokenizer import BPETokenizer

# downloads part of TinyStories, trains the tokenizer, and writes token ids to .bin files
# that train.py reads with np.memmap, so the whole dataset never has to sit in memory

DATA_DIR = "data"
BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
VAL_FILE = "TinyStoriesV2-GPT4-valid.txt"

# the full train file is 2.2GB; the first 300MB is still far more tokens than we train on
TRAIN_BYTES = 300_000_000

VOCAB_SIZE = 2048
TOKENIZER_TRAIN_CHARS = 5_000_000
EOT = "<|endoftext|>"

def download(name, max_bytes=None):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return path
    req = urllib.request.Request(BASE_URL + name)
    if max_bytes is not None:
        # HTTP range request: only fetch the start of the file
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    print(f"Downloading {name}...")
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    return path

def read_stories(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    stories = [s.strip() for s in text.split(EOT)]
    # a partial download can cut the last story off mid-sentence, so drop it
    if not text.rstrip().endswith(EOT):
        stories = stories[:-1]
    return [s for s in stories if s]

def write_tokens(tok, stories, out_path):
    eot_id = tok.special_tokens[EOT]
    t0 = time.time()
    buffers, n_tokens = [], 0
    with open(out_path, "wb") as f:
        for i, story in enumerate(stories):
            # every story ends with <|endoftext|> so the model learns where stories stop
            buffers.extend(tok.encode(story))
            buffers.append(eot_id)
            if len(buffers) > 1_000_000:
                np.array(buffers, dtype=np.uint16).tofile(f)
                n_tokens += len(buffers)
                buffers = []
            if (i + 1) % 100_000 == 0:
                print(f"  {i + 1:,}/{len(stories):,} stories | {n_tokens:,} tokens | "
                      f"{time.time() - t0:.0f}s")
        np.array(buffers, dtype=np.uint16).tofile(f)
        n_tokens += len(buffers)
    return n_tokens

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = download(TRAIN_FILE, TRAIN_BYTES)
    val_path = download(VAL_FILE)

    train_stories = read_stories(train_path)
    val_stories = read_stories(val_path)
    print(f"{len(train_stories):,} train stories, {len(val_stories):,} val stories")

    # BPE training is slow in pure Python, so learn the merges from a sample
    sample, n_chars = [], 0
    for story in train_stories:
        sample.append(story)
        n_chars += len(story)
        if n_chars >= TOKENIZER_TRAIN_CHARS:
            break

    print(f"\nTraining BPE tokenizer (vocab_size={VOCAB_SIZE}) on {n_chars:,} chars...")
    tok = BPETokenizer()
    # reserve one id for <|endoftext|>
    tok.train("\n\n".join(sample), num_merges=VOCAB_SIZE - 256 - 1)
    tok.add_special_token(EOT)
    tok.save(os.path.join(DATA_DIR, "tokenizer.json"))
    # uint16 storage holds ids up to 65535
    assert tok.vocab_size <= 2**16
    print(f"Tokenizer saved. Vocab size: {tok.vocab_size}")

    for split, stories in (("val", val_stories), ("train", train_stories)):
        print(f"\nEncoding {split}...")
        n_chars = sum(len(s) for s in stories)
        n_tokens = write_tokens(tok, stories, os.path.join(DATA_DIR, f"{split}.bin"))
        print(f"{split}: {n_chars:,} chars -> {n_tokens:,} tokens "
              f"({n_chars / n_tokens:.2f} chars/token)")
