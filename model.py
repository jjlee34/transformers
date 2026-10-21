import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=1024, dropout=0.0):
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

        # scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # apply causal mask
        attn_scores = attn_scores.masked_fill(~self.causal_mask[:T, :T], float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v

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

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=1024, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)
    def forward(self, x):
        # pre-norm and residual to normalize before the sublayer
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d_model, n_heads = 2, 16, 256, 8

    block = TransformerBlock(d_model, n_heads, max_seq_len=T)
    x = torch.randn(B, T, d_model)
    out = block(x)

    print("input shape: ", x.shape)
    print("output shape: ", out.shape)

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 max_seq_len=256, dropout=0.0):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
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

        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
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
        return idx

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d_model, n_heads = 2, 16, 256, 8

    block = TransformerBlock(d_model, n_heads, max_seq_len=T)
    x = torch.randn(B, T, d_model)
    out = block(x)

    print("input shape: ", x.shape)
    print("output shape:", out.shape)