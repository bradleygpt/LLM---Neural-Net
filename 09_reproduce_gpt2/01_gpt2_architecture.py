"""
01_gpt2_architecture.py
-----------------------
The transformer you built in lesson 07 is essentially correct. To match
the published GPT-2 model exactly, we need a small number of refinements.
None of them are conceptual surprises -- they're engineering details
that the original GPT-2 paper specified.

The differences:

  1. SIZE. GPT-2 small (124M params) uses:
        block_size = 1024  (vs lesson 07's 256)
        n_embed    = 768   (vs 384)
        n_heads    = 12    (vs 6)
        n_layers   = 12    (vs 6)
        vocab_size = 50,257 (real GPT-2 BPE vocab, vs 65 character)

  2. GELU instead of ReLU in the feed-forward layers. GPT-2 uses
     the original "tanh approximation" of GELU. We match that exactly.

  3. WEIGHT TYING. The token embedding (50257 -> 768) and the language
     modeling head (768 -> 50257) share weights -- they're literally
     the same tensor. This saves 38M parameters (768*50257 = 38M)
     and helps training stability. Lesson 07 kept them separate.

  4. SPECIFIC WEIGHT INITIALIZATION:
       - All linear/embedding weights: N(0, 0.02)
       - All biases: zeros
       - SPECIAL: Linear weights inside residual paths get scaled
         by 1/sqrt(2*n_layers). This compensates for the variance
         growth from N residual additions, keeping activations stable.

  5. Slight LayerNorm placement difference (which we already do correctly):
     pre-LN: residual = x + sublayer(LN(x)). GPT-2 uses pre-LN. Good.

This script just BUILDS the architecture and reports param counts. It
doesn't load weights or train -- those come in the next scripts.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------
class GPTConfig:
    """Configuration for GPT-2 small (124M)."""
    block_size: int = 1024
    vocab_size: int = 50257       # real GPT-2 BPE vocab
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0          # GPT-2 paper used 0.1; for fine-tuning we'll use 0.0


# ---------------------------------------------------------------
# Causal self-attention -- multi-head, scaled dot-product, masked
# ---------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention. All heads computed in one
    fused matmul for efficiency (vs the loop-of-heads in lesson 07)."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # One fused linear that produces Q, K, V together.
        # Output dim = 3 * n_embd: first n_embd is Q, next K, last V.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection back to n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Causal mask (registered as a buffer, not a parameter)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        # Calculate Q, K, V in one matmul
        qkv = self.c_attn(x)                                # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)             # each (B, T, C)
        # Reshape for multi-head: (B, T, n_head, head_dim) -> transpose -> (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))   # (B, n_head, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                          # (B, n_head, T, head_dim)
        # Re-assemble heads back to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


# ---------------------------------------------------------------
# Feed-forward (the MLP block in each transformer layer)
# ---------------------------------------------------------------
class MLP(nn.Module):
    """Two-layer MLP with GELU (tanh approximation) in between.

    The 4x expansion (n_embd -> 4*n_embd -> n_embd) is standard GPT-2.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GPT-2 uses the *tanh approximation* of GELU.
        # Modern GPT-2 reproductions sometimes use the exact GELU instead;
        # they're nearly identical. We match the original paper.
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


# ---------------------------------------------------------------
# Transformer block (attention + ffn, both wrapped in residual+LN)
# ---------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN with residuals (same as lesson 07)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------
# The full GPT-2 model
# ---------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Module dict matches HuggingFace's naming, so weight-loading is easy.
        # wte = "word token embedding"
        # wpe = "word position embedding"
        # h   = "hidden" (the stack of transformer blocks)
        # ln_f = final layer norm
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # Language modeling head: project hidden states back to vocab logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # WEIGHT TYING: the token embedding and the lm_head share weights.
        # This is a single tensor that gets used both for input lookup and
        # output projection. Saves 38M parameters AND helps training.
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Special scaled init for residual-path weights
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                # Compensate for variance growth from N residual additions.
                # Without this, the network's activations grow unboundedly with depth.
                std = 0.02 / math.sqrt(2 * config.n_layer)
                torch.nn.init.normal_(p, mean=0.0, std=std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)               # (T, n_embd)
        tok_emb = self.transformer.wte(idx)               # (B, T, n_embd)
        x = tok_emb + pos_emb
        # Pass through stack of blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss


# ---------------------------------------------------------------
# Build the model and report
# ---------------------------------------------------------------
if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)

    # Count parameters. PyTorch's model.parameters() automatically dedupes
    # shared tensors (so the tied wte/lm_head weight is counted once), giving
    # us the canonical published parameter count.
    n_params = sum(p.numel() for p in model.parameters())

    print(f"GPT-2 small architecture built.")
    print(f"  block_size:  {config.block_size}")
    print(f"  vocab_size:  {config.vocab_size}")
    print(f"  n_layer:     {config.n_layer}")
    print(f"  n_head:      {config.n_head}")
    print(f"  n_embd:      {config.n_embd}")
    print()
    print(f"Total parameters: {n_params / 1e6:.2f}M")
    print(f"Published GPT-2 small: 124M")
    print(f"Match: {abs(n_params / 1e6 - 124) < 1.0}")
    print()

    # Quick forward pass
    print("Sanity check forward pass:")
    idx = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(idx, targets=idx)
    print(f"  Input shape:   {tuple(idx.shape)}")
    print(f"  Logits shape:  {tuple(logits.shape)}")
    print(f"  Loss (random): {loss.item():.4f}")
    print(f"  Expected (log V = {math.log(config.vocab_size):.2f}): close to that")


# ---------------------------------------------------------------
# What we just built
# ---------------------------------------------------------------
"""
============================================================
THE ARCHITECTURE OF GPT-2 SMALL (124M)
============================================================

Six refinements from your lesson 07 transformer:

  1. Bigger numbers (1024 ctx, 768 dim, 12 layers, 12 heads, 50257 vocab)
  2. Fused QKV in attention (one 3x matmul instead of 3 separate)
  3. GELU (tanh approx) instead of ReLU in feedforward
  4. Weight tying: token embedding == lm_head weights (same tensor)
  5. Scaled init for residual-path output projections
  6. Module names match HuggingFace's so weight loading is trivial

Total: 124M parameters. Same architecture OpenAI published in 2019.

Next: download and load OpenAI's actual published weights into this model.
"""
