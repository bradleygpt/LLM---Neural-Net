"""
06_full_gpt.py
--------------
Assemble the full GPT. The architecture:

    tokens (B, T)
      |
      | token embedding lookup
      v
    + position embedding lookup
      v
    embed (B, T, C)
      |
      | repeat N transformer blocks:
      |   Block: x + attention(LN(x)), then x + ffn(LN(x))
      v
    final LayerNorm
      v
    Linear -> logits (B, T, V)

Two important details:

  1. POSITION EMBEDDING. Attention itself is permutation-invariant --
     it doesn't know that position 3 comes before position 5. We add
     a learnable position embedding to fix this. Each position 0..T-1
     gets its own learnable C-dim vector. (Real GPT-3 uses learned
     position embeddings; some newer models use rotary or relative
     position embeddings.)

  2. WEIGHT TIE OPTIONAL. Many GPT-style models tie the input token
     embedding's weights to the output Linear's weights (they're
     transposed views of the same matrix). Saves params, often helps.
     We don't tie here for clarity; lesson 09 (GPT-2 reproduction)
     does tie them.

This script just builds the model and verifies shapes. The next script
trains it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Building blocks (copy of script 05 for self-containment)
# ---------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x); v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embed, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_heads, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ---------------------------------------------------------------
# The full GPT
# ---------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embed, n_heads, n_layers, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[
            Block(n_embed, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Token + position embeddings, summed
        tok = self.token_emb(idx)                                  # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))      # (T, C)
        x = tok + pos                                              # broadcast to (B, T, C)
        x = self.blocks(x)                                         # (B, T, C)
        x = self.ln_f(x)                                           # final LN
        logits = self.lm_head(x)                                   # (B, T, V)

        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new):
        for _ in range(max_new):
            # Crop context to block_size (model can't see further)
            idx_crop = idx[:, -self.block_size:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


# ---------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------
torch.manual_seed(1337)

# Tiny config to verify the architecture works
V, n_embed, n_heads, n_layers, block_size = 65, 64, 4, 4, 8

m = GPT(V, n_embed, n_heads, n_layers, block_size)
print(f"Tiny GPT parameters: {sum(p.numel() for p in m.parameters()):,}")
print()

# Try a forward pass
B, T = 4, block_size
idx = torch.randint(0, V, (B, T))
targets = torch.randint(0, V, (B, T))
logits, loss = m(idx, targets)
print(f"Input shape:   {tuple(idx.shape)}")
print(f"Logits shape:  {tuple(logits.shape)}")
print(f"Initial loss:  {loss.item():.4f}")
import math
print(f"Expected (random init, log V): {math.log(V):.4f}")
print()

# Try generating
print("Untrained generation (should be garbage):")
out = m.generate(torch.zeros((1, 1), dtype=torch.long), max_new=30)
print(f"  shape: {tuple(out.shape)}")
print(f"  first 30 token indices: {out[0].tolist()}")
print()


# ---------------------------------------------------------------
# Architecture summary
# ---------------------------------------------------------------
print("=" * 60)
print("THE COMPLETE GPT, IN ONE EXPRESSION")
print("=" * 60)
print("""
GPT(idx) =
    LayerNorm_final(
        Block_N(...
            Block_2(
                Block_1(
                    TokenEmbed(idx) + PositionEmbed(0..T-1)
                )
            )
        ...)
    ) -> Linear -> logits

Each Block:
    x = x + MultiHeadAttention(LayerNorm(x))   # communicate
    x = x + FeedForward(LayerNorm(x))          # compute

That's it. That's the entire architecture that powers ChatGPT, Claude,
Gemini, and every other modern LLM. The differences at scale are:
  - Bigger numbers (more layers, larger embedding dim, more heads)
  - Different position embeddings (RoPE, ALiBi, etc.)
  - Various attention optimizations (flash attention, grouped query)
  - Trained on TRILLIONS of tokens instead of one Shakespeare file

Architecturally, you've now built it.
""")
