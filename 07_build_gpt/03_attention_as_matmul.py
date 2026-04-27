"""
03_attention_as_matmul.py
-------------------------
Before we touch the full attention mechanism, walk through the
MATHEMATICAL TRICK at its core. This is the single most important
insight in the whole transformer paper.

The bigram model only sees the previous token. We want each token to
see ALL previous tokens (causal context). The naive way: for each
position t, compute a weighted average of tokens 0..t. Weights tell
the model "how much should position t pay attention to position 3?"

The naive way is slow. Karpathy's trick: this weighted average is just
a MATRIX MULTIPLICATION with a triangular matrix.

We'll demonstrate this in three escalating forms:

  v1: For loop -- explicit, slow, easy to read.
  v2: Matrix multiplication with a uniform triangular weight matrix.
       Same answer, ~1000x faster.
  v3: Use softmax over a triangular mask so weights can be LEARNED
       instead of uniform. This is the actual attention pattern.

After this script you'll see why attention is so fast on GPUs: it's
just one matmul.
"""

import torch
import torch.nn.functional as F


torch.manual_seed(1337)
B, T, C = 4, 8, 2  # batch, time, channels (features per token)
x = torch.randn(B, T, C)
print(f"x shape: {tuple(x.shape)}  (B, T, C)")
print()


# ---------------------------------------------------------------
# v1: For loop (the naive version)
# ---------------------------------------------------------------
# At position t, average all tokens 0..t. So the output at position t
# is the running mean of the input from position 0 to t.
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]            # (t+1, C)
        xbow[b, t] = xprev.mean(dim=0)  # (C,)

print("v1 (for loop):")
print(f"  output shape: {tuple(xbow.shape)}")
print(f"  xbow[0]:\n{xbow[0]}")
print()


# ---------------------------------------------------------------
# v2: Matrix multiplication with a triangular weight matrix
# ---------------------------------------------------------------
# Build a (T, T) lower-triangular matrix where each row sums to 1:
#
#     1.00 0.00 0.00 0.00 ...
#     0.50 0.50 0.00 0.00 ...
#     0.33 0.33 0.33 0.00 ...
#     0.25 0.25 0.25 0.25 ...
#
# Multiply: weights @ x  -- for each batch this gives the running mean.

wei = torch.tril(torch.ones(T, T))                # lower-triangular ones
wei = wei / wei.sum(dim=1, keepdim=True)          # normalize rows
print("v2 weight matrix (each row averages tokens 0..t):")
print(wei)
print()

# Apply to x: (T, T) @ (B, T, C) -> (B, T, C) (batched matmul)
xbow2 = wei @ x
print(f"v2 output shape: {tuple(xbow2.shape)}")
print(f"v1 vs v2 maximum difference: {(xbow - xbow2).abs().max().item():.2e}")
print(f"  -> they're equal. The matmul replaces the for loop.")
print()


# ---------------------------------------------------------------
# v3: Softmax over a triangular mask (the attention form)
# ---------------------------------------------------------------
# Instead of uniform weights, start with arbitrary "attention scores"
# and softmax them. The triangular mask -inf prevents future tokens
# from contributing.
tril = torch.tril(torch.ones(T, T))
wei3 = torch.zeros((T, T))                       # all-zero attention scores
wei3 = wei3.masked_fill(tril == 0, float("-inf"))  # -inf above diagonal
wei3 = F.softmax(wei3, dim=-1)
print("v3 weight matrix (softmax over triangular mask):")
print(wei3)
print(f"  (since scores are all 0, softmax of [0,0,...,0,-inf,-inf] gives uniform")
print(f"   weights over the valid positions -- same as v2)")
print()

xbow3 = wei3 @ x
print(f"v3 output shape: {tuple(xbow3.shape)}")
print(f"v1 vs v3 maximum difference: {(xbow - xbow3).abs().max().item():.2e}")
print(f"  -> all three give the same answer.")
print()


# ---------------------------------------------------------------
# So why is v3 useful?
# ---------------------------------------------------------------
# In v2 we used UNIFORM weights (each previous token equally important).
# In v3 we computed weights with softmax over -- so far -- all-zero scores.
#
# The leap to ATTENTION: make those scores LEARNABLE. Compute them from
# the input itself. Some tokens will then contribute MORE than others
# to a given position's output.
#
# That's it. Attention = "weighted average where the weights are a
# learned function of the input." The triangular mask keeps it CAUSAL
# (can't see the future).
#
# In v3 the scores were a constant zero matrix. In real attention,
# the scores are computed from the input via:
#
#     scores[i, j] = (query at position i) DOT (key at position j) / sqrt(C)
#
# We build that next.

print("=" * 60)
print("THE INSIGHT")
print("=" * 60)
print("""
A weighted average of past tokens is a matrix multiplication.
A triangular mask makes it causal (no peeking ahead).
Replacing UNIFORM weights with LEARNABLE weights gives self-attention.
That's the entire conceptual leap from WaveNet's fixed merging to
the transformer's learned merging.
""")
