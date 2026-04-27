"""
05_multi_head_and_block.py
--------------------------
Two more pieces complete the transformer block:

  1. MULTI-HEAD ATTENTION. Run several attention "heads" in parallel,
     concatenate their outputs. Each head can specialize in a different
     kind of relationship (subjects-verbs, modifiers-nouns, etc.).
     With H total channels and N heads, each head has H/N channels.

  2. FEED-FORWARD NETWORK. After attention "communicates" between
     positions, an MLP applied to each position INDEPENDENTLY lets the
     model "think" about the gathered information. It's a Linear ->
     ReLU -> Linear with a 4x widening factor in the middle (standard
     transformer convention).

Plus two structural details that make deep transformers actually train:

  3. RESIDUAL CONNECTIONS (skip connections). The output of attention
     and the output of the feed-forward are ADDED to their inputs, not
     replacing them. This creates a "highway" gradients can flow
     backward along, even through many stacked blocks.

  4. LAYER NORM. Normalize across the feature dimension (per token,
     per example). Like BatchNorm but does NOT couple examples in a
     batch. That's why transformers use LayerNorm: same formula at
     training and inference time, no running stats needed.

A "transformer block" = LayerNorm + MultiHeadAttention + Residual,
                       LayerNorm + FeedForward + Residual.

Stack N of these blocks and you have GPT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Single attention head (refactored from script 04 into a class)
# ---------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # Save the lower-triangular mask as a buffer (not a learned parameter)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)        # (B, T, head_size)
        q = self.query(x)
        v = self.value(x)
        # scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)   # (B, T, T)
        # causal mask -- truncate tril to current sequence length
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v          # (B, T, head_size)


# ---------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embed, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embed, head_size, block_size, dropout) for _ in range(n_heads)
        ])
        # After concat, project back to n_embed (so the residual addition
        # is dimensionally consistent)
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate along the channel dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------
# Feed-forward network (the "MLP" in a transformer block)
# ---------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        # Standard transformer convention: 4x expansion in the middle
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------
# Transformer block: communication then computation
# ---------------------------------------------------------------
class Block(nn.Module):
    """One transformer block.

    Pre-LN convention: LayerNorm BEFORE attention/FFN, residual outside.
    This is what GPT-2 and later models use. The original "Attention Is
    All You Need" paper used post-LN; pre-LN trains more stably.
    """
    def __init__(self, n_embed, n_heads, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Residual: x + (sublayer applied to LN(x))
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ---------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------
torch.manual_seed(1337)

n_embed = 64        # embedding dim
n_heads = 4         # number of attention heads (each gets 64/4 = 16 dims)
block_size = 8      # max context length
B, T = 4, block_size

x = torch.randn(B, T, n_embed)
print(f"Input: {tuple(x.shape)}")

# One head
h = Head(n_embed, head_size=16, block_size=block_size)
print(f"Single Head output: {tuple(h(x).shape)}")

# Multi-head
mha = MultiHeadAttention(n_heads, head_size=16, n_embed=n_embed, block_size=block_size)
print(f"MultiHeadAttention output: {tuple(mha(x).shape)}")

# Feed-forward
ffn = FeedForward(n_embed)
print(f"FeedForward output: {tuple(ffn(x).shape)}")

# Full block
block = Block(n_embed, n_heads, block_size)
print(f"Block output: {tuple(block(x).shape)}")
print()
print(f"Block parameter count: {sum(p.numel() for p in block.parameters()):,}")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
print("""
A transformer block has two "phases":

  1. COMMUNICATION (multi-head attention):
     Each token gathers information from previous tokens via attention.
     With multiple heads, the model can attend to different "aspects"
     in parallel.

  2. COMPUTATION (feed-forward):
     Applied per-position, independently. After gathering info from
     other positions, each position now "thinks" about what it gathered.

The residual + LayerNorm pattern around each phase is what makes deep
stacks of these trainable. Without residuals, gradients struggle to
flow back through 6+ blocks. With residuals, you can stack 96 (GPT-3).

Next: assemble these blocks into a full GPT and train it.
""")
