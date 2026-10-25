import json
from collections import Counter
import regex as re

GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# pre-processing: turns the text into list of byte-tuples
def pretokenize(text):
    chunks = re.findall(GPT2_SPLIT_PATTERN, text)
    return [tuple(chunk.encode("utf-8")) for chunk in chunks]

# pre-processing: calculates pair counts using chunk frequency; simple version
def get_pair_counts(chunk_freq):
    pair_counts = Counter()
    for chunk, freq in chunk_freq.items():
        for i in range(len(chunk) - 1):
            pair_counts[(chunk[i], chunk[i + 1])] += freq
    return pair_counts

# pre-processing: replaces every occurence of a given pair inside a chunk with a new token id
def merge_pair(chunk, pair, new_id):
    merged = []
    i = 0
    while i < len(chunk):
        if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == pair:
            merged.append(new_id)
            i+=2
        else:
            merged.append(chunk[i])
            i+=1
    return tuple(merged)

# training: breaks down text into chunks and performs merges based on pair count frequency
def train_bpe(text, num_merges):
    chunk_freq = Counter(pretokenize(text))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}
    next_id = 256
    for step in range(num_merges):
        pair_counts = get_pair_counts(chunk_freq)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)

        new_chunk_freq = Counter()
        for chunk, freq in chunk_freq.items():
            new_chunk_freq[merge_pair(chunk, best_pair, next_id)] += freq
        chunk_freq = new_chunk_freq

        merges[best_pair] = next_id
        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        next_id += 1

        if (step + 1) % 50 == 0 or step == num_merges - 1:
            print(f" merges {step + 1}/{num_merges}: {best_pair} -> {next_id - 1} "
                  f"({vocab[next_id - 1]})")
    return merges, vocab

class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        # special tokens like <|endoftext|> get their own id and are never split by BPE
        self.special_tokens = {}
        # chunk -> token ids; text repeats the same words constantly, so this saves re-merging
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, num_merges):
        self.merges, self.vocab = train_bpe(text, num_merges)
        self._cache = {}

    def add_special_token(self, token):
        new_id = len(self.vocab)
        self.special_tokens[token] = new_id
        self.vocab[new_id] = token.encode("utf-8")
        return new_id

    # mirrors the training process and outputs token ids for the text based on merges
    def encode(self, text):
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            token_ids = []
            for part in re.split(pattern, text):
                if part in self.special_tokens:
                    token_ids.append(self.special_tokens[part])
                elif part:
                    token_ids.extend(self._encode_ordinary(part))
            return token_ids
        return self._encode_ordinary(text)

    def _encode_ordinary(self, text):
        chunks = pretokenize(text)
        token_ids = []
        for chunk in chunks:
            cached = self._cache.get(chunk)
            if cached is not None:
                token_ids.extend(cached)
                continue
            tokens = list(chunk)
            while len(tokens) >= 2:
                pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                mergable = [p for p in pairs if p in self.merges]
                if not mergable:
                    break
                pair_to_merge = min(mergable, key=lambda p : self.merges[p])
                new_id = self.merges[pair_to_merge]
                tokens = list(merge_pair(tuple(tokens), pair_to_merge, new_id))
            self._cache[chunk] = tokens
            token_ids.extend(tokens)
        return token_ids

    # turns a list of token ids back into text
    def decode(self, token_ids):
        byte_seq = b"".join(self.vocab[t] for t in token_ids)
        return byte_seq.decode("utf-8", errors="replace")

    def save(self, path):
        data = {
            "merges": [[list(pair), new_id] for pair, new_id in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in data["merges"]:
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.special_tokens = data.get("special_tokens", {})
        for token, new_id in self.special_tokens.items():
            self.vocab[new_id] = token.encode("utf-8")
        self._cache = {}