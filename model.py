import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# modern (Llama-style) replacements, each switched on by an option in GPT:
#   norm="rmsnorm"  -> RMSNorm instead of LayerNorm
#   pos="rope"      -> rotary position embeddings instead of a learned pos_emb table
#   mlp="swiglu"    -> gated SwiGLU MLP instead of the GELU MLP
# the defaults ("layernorm", "learned", "gelu") are the original GPT-2 model

class RMSNorm(nn.Module):
    # LayerNorm subtracts the mean, divides by the std, then scales and shifts.
    # RMSNorm skips the mean and the shift: it only divides by the root-mean-square
    # and scales. cheaper, and works as well in practice
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

def make_norm(norm, d_model):
    if norm == "layernorm":
        return nn.LayerNorm(d_model)
    if norm == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"unknown norm {norm!r}")

class Rotary(nn.Module):
    # RoPE: instead of adding a position vector to the token embedding, rotate q and k
    # by an angle that grows with position. each pair of dimensions (0,1), (2,3), ...
    # rotates at its own speed. the dot product q_i . k_j then depends only on the
    # distance i - j, so attention sees relative position directly
    def __init__(self, d_head, max_seq_len, base=10000):
        super().__init__()
        # one frequency per dimension pair: fast for the first pairs, slow for the last
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        # angle for every (position, pair): (max_seq_len, d_head / 2)
        angles = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        # x: (B, n_heads, T, d_head)
        T = x.shape[-2]
        cos, sin = self.cos[:T], self.sin[:T]
        # split each vector into its pairs and rotate every pair by its angle
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=1024, dropout=0.0, rope=False):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # one combined projection for q, k, v for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # causal mask to enforce sequential training and prediction
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.rotary = Rotary(self.d_head, max_seq_len) if rope else None

    def forward(self, x):
        # batch, sequence length, and d_model
        B, T, C = x.shape

        # (B, T, 3*C)
        qkv = self.qkv_proj(x)
        # each (B, T, C)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # splitting d_model into different attention heads
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # RoPE puts position into q and k only; v carries content and is left alone
        if self.rotary is not None:
            q = self.rotary(q)
            k = self.rotary(k)

        # scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # apply causal mask
        attn_scores = attn_scores.masked_fill(~self.causal_mask[:T, :T], float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v

        # the four steps above can be replaced by one fused call. it computes the
        # same thing without storing the (T, T) score matrix; on CUDA it uses
        # FlashAttention and is much faster, but on MPS it measured ~5% slower here.
        # it scales by 1/sqrt(d_head) itself, and is_causal=True applies the mask,
        # so the causal_mask buffer in __init__ would no longer be needed.
        # out = F.scaled_dot_product_attention(
        #     q, k, v,
        #     is_causal=True,
        #     dropout_p=self.dropout.p if self.training else 0.0,
        # )

        # merge heads back into (B, T, C)
        out = out.transpose(1, 2). contiguous().view(B, T, C)
        return self.out_proj(out)

class MLP(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.0):
        super().__init__()
        d_hidden = expansion * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SwiGLU(nn.Module):
    # gated MLP: one projection (gate) decides how much of another (up) to let through,
    # per hidden unit. SiLU(x) = x * sigmoid(x) is a smooth GELU-like activation.
    # three matrices instead of two, so the hidden size is 8/3 * d_model rather than
    # 4 * d_model to keep the parameter count about the same as the GELU MLP
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        d_hidden = int(8 * d_model / 3)
        self.gate = nn.Linear(d_model, d_hidden)
        self.up = nn.Linear(d_model, d_hidden)
        self.down = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=1024, dropout=0.0,
                 norm="layernorm", rope=False, mlp="gelu"):
        super().__init__()
        self.ln1 = make_norm(norm, d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, max_seq_len, dropout, rope)
        self.ln2 = make_norm(norm, d_model)
        if mlp == "gelu":
            self.mlp = MLP(d_model, dropout=dropout)
        elif mlp == "swiglu":
            self.mlp = SwiGLU(d_model, dropout=dropout)
        else:
            raise ValueError(f"unknown mlp {mlp!r}")
    def forward(self, x):
        # pre-norm and residual to normalize before the sublayer
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 max_seq_len=256, dropout=0.0,
                 norm="layernorm", pos="learned", mlp="gelu"):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        # with RoPE, position goes into attention instead, so there is no pos_emb table
        if pos == "learned":
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
        elif pos == "rope":
            self.pos_emb = None
        else:
            raise ValueError(f"unknown pos {pos!r}")
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, dropout,
                             norm=norm, rope=(pos == "rope"), mlp=mlp)
            for _ in range(n_layers)
        ])

        self.ln_f = make_norm(norm, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # B: batch size, T: length of sequences (context window)
        B, T = idx.shape
        assert T <= self.max_seq_len, f"sequence lenght {T} exceeds max_seq_len {self.max_seq_len}"

        x = self.token_emb(idx)
        if self.pos_emb is not None:
            positions = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eot_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            # stop once every sequence in the batch has produced <|endoftext|>
            if eot_id is not None and (next_id == eot_id).all():
                break
        return idx

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d_model, n_heads = 2, 16, 256, 8

    block = TransformerBlock(d_model, n_heads, max_seq_len=T)
    x = torch.randn(B, T, d_model)
    out = block(x)

    print("input shape: ", x.shape)
    print("output shape:", out.shape)