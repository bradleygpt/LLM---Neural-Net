"""
04_self_attention.py
--------------------
Build self-attention from scratch.

The previous script showed that "weighted average of past tokens"
collapses into one matmul. The weights were uniform. Now we make them
LEARNABLE, computed from the input itself.

The recipe (this is the entire transformer, in 5 lines):

  1. Each token at position i emits a QUERY vector q_i: "what am I looking for?"
  2. Each token at position j emits a KEY vector k_j: "what do I contain?"
  3. The attention WEIGHT for position i looking at position j is:
        weight[i, j] = q_i DOT k_j  (high when query and key align)
     Then we softmax across j to make the weights sum to 1, and apply a
     CAUSAL MASK so position i can only see positions 0..i.
  4. Each token also emits a VALUE vector v_j: "if you attend to me,
     here's what to take."
  5. The output at position i is the weighted sum: sum_j weight[i, j] * v_j.

Q, K, V are produced by three separate Linear layers. Their inputs are
the same tensor (the token embeddings); their weights are different.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
torch.manual_seed(1337)
B, T, C = 4, 8, 32   # batch, time, channels
x = torch.randn(B, T, C)
print(f"x shape: {tuple(x.shape)}  (B, T, C)")
print()


# ---------------------------------------------------------------
# Single-head self-attention
# ---------------------------------------------------------------
HEAD = 16  # dimension of each attention head

# Three linear projections: input C -> HEAD
key = nn.Linear(C, HEAD, bias=False)
query = nn.Linear(C, HEAD, bias=False)
value = nn.Linear(C, HEAD, bias=False)

# Apply them. Each gives (B, T, HEAD).
k = key(x)
q = query(x)
v = value(x)
print(f"k shape: {tuple(k.shape)}, q shape: {tuple(q.shape)}, v shape: {tuple(v.shape)}")
print()


# ---------------------------------------------------------------
# Compute attention weights
# ---------------------------------------------------------------
# wei[b, i, j] = q[b, i, :] DOT k[b, j, :]
#   -> Compute via batched matmul: q @ k.transpose(-2, -1)
#   -> q is (B, T, H), k is (B, T, H), k.transpose(-2,-1) is (B, H, T)
#   -> result: (B, T, T)
wei = q @ k.transpose(-2, -1)
print(f"raw attention scores shape: {tuple(wei.shape)}  (B, T, T)")
print(f"raw scores stats: mean={wei.mean().item():.3f}, std={wei.std().item():.3f}")
print()


# ---------------------------------------------------------------
# Scale by sqrt(HEAD)
# ---------------------------------------------------------------
# Without this, dot products grow with HEAD's magnitude, pushing softmax
# into a near-one-hot regime (saturated). Scaling keeps variance ~1.
wei = wei * (HEAD ** -0.5)
print(f"After scaling by 1/sqrt(HEAD={HEAD}):")
print(f"  scores stats: mean={wei.mean().item():.3f}, std={wei.std().item():.3f}")
print(f"  -> std now ~1, softmax will produce moderate weights instead of saturating.")
print()


# ---------------------------------------------------------------
# Apply causal mask
# ---------------------------------------------------------------
# Set everything above the diagonal to -inf so future tokens can't
# influence the current position.
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))


# ---------------------------------------------------------------
# Softmax to get probabilities
# ---------------------------------------------------------------
wei = F.softmax(wei, dim=-1)
print("Attention weights, batch element 0 (each row should sum to 1):")
print(wei[0])
print()


# ---------------------------------------------------------------
# Aggregate values
# ---------------------------------------------------------------
# out[b, i, :] = sum_j wei[b, i, j] * v[b, j, :]
# Batched matmul: (B, T, T) @ (B, T, H) -> (B, T, H)
out = wei @ v
print(f"output shape: {tuple(out.shape)}")
print()


# ---------------------------------------------------------------
# Notice what we just built
# ---------------------------------------------------------------
print("=" * 60)
print("WHAT JUST HAPPENED")
print("=" * 60)
print("""
Each token at position i now produces an output vector that is a
weighted sum of the value vectors of tokens 0..i. The weights are
computed from the dot product of i's query with each j's key.

If a query at position 5 has high alignment with the key at position 2,
then position 5's output will heavily incorporate position 2's value.
The model LEARNS what queries, keys, and values look like during
training -- by adjusting the weights of the three Linear layers.

Compare to WaveNet: WaveNet's "merge" was always pairs of adjacent
tokens. Attention's "merge" is a learned weighted combination of ALL
previous tokens. That flexibility is what makes transformers so
much more capable.

A few subtle points worth noting:

  - Q, K, V are computed from the SAME input x. That's why this is
    "SELF-attention" -- the sequence is attending to itself. (Cross-
    attention, used in encoder-decoder models, has Q from one sequence
    and K, V from another.)

  - The bias=False on the Linears is conventional. Biases would just
    add a constant to every dot product, which doesn't change the
    softmax weights.

  - The 1/sqrt(HEAD) scaling is from the original "Attention Is All
    You Need" paper. Skipping it makes training unstable.

Next: bundle this into a class and stack multiple heads.
""")
