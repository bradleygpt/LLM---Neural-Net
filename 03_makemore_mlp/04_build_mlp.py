"""
04_build_mlp.py
---------------
Build the full Bengio 2003 architecture, forward pass only.

Architecture:

    indices (N, 3)
        |
        | embedding lookup C[indices]
        v
    embeddings (N, 3, D)
        |
        | view/reshape: flatten the context dim
        v
    flat_emb (N, 3*D)
        |
        | linear layer: @W1 + b1, then tanh
        v
    hidden (N, hidden_size)
        |
        | linear layer: @W2 + b2
        v
    logits (N, V)

The trickiest line is the .view() that flattens the (N, 3, D) embeddings
into (N, 3*D). The 3 context positions get CONCATENATED into one long
vector per example. Order matters: the first context position's
D-vector ends up first in the flat row, etc.

We use TANH for the hidden layer activation -- same as the original
paper, and it'll come back to bite us in lesson 04 (saturated tanh
neurons kill gradients).
"""

import torch
import torch.nn.functional as F


# Load data
data = torch.load("dataset.pt", weights_only=False)
Xtr, Ytr = data["Xtr"], data["Ytr"]
V = len(data["itos"])
BLOCK = data["block_size"]


# ---------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------
D = 10                # embedding dimension
HIDDEN = 200          # hidden layer size

torch.manual_seed(2147483647)


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
C = torch.randn((V, D))                         # embedding table
W1 = torch.randn((BLOCK * D, HIDDEN)) * 0.1      # hidden layer weights (small init)
b1 = torch.randn(HIDDEN) * 0.1                  # hidden layer biases
W2 = torch.randn((HIDDEN, V)) * 0.1              # output layer weights
b2 = torch.randn(V) * 0.1                       # output layer biases

parameters = [C, W1, b1, W2, b2]
total = sum(p.numel() for p in parameters)
print(f"Parameters: {total}")
for name, p in zip(["C", "W1", "b1", "W2", "b2"], parameters):
    print(f"  {name}: {tuple(p.shape)} = {p.numel():,}")
print()


# ---------------------------------------------------------------
# Forward pass on the first 32 training examples (a "minibatch")
# ---------------------------------------------------------------
ix = torch.arange(32)  # just take the first 32 examples for a demo
Xb = Xtr[ix]   # (32, 3)
Yb = Ytr[ix]   # (32,)

# 1. Embed
emb = C[Xb]                                  # (32, 3, D)
print(f"emb.shape = {tuple(emb.shape)}")

# 2. Flatten the context dimension
# .view(N, -1) tells PyTorch "keep N rows, infer the other dimension".
# It's equivalent to .view(32, 30) here since 3*10 = 30.
flat = emb.view(emb.shape[0], -1)            # (32, 30)
print(f"flat.shape = {tuple(flat.shape)}")

# 3. Hidden layer
h = torch.tanh(flat @ W1 + b1)               # (32, HIDDEN)
print(f"h.shape = {tuple(h.shape)}")

# 4. Output layer
logits = h @ W2 + b2                         # (32, V)
print(f"logits.shape = {tuple(logits.shape)}")
print()


# ---------------------------------------------------------------
# Compute loss
# ---------------------------------------------------------------
# We could compute softmax + NLL by hand (script 06 of the bigram lesson),
# but PyTorch has a single function that does both with better numerical
# stability: F.cross_entropy. It expects RAW LOGITS, not probabilities.
loss = F.cross_entropy(logits, Yb)
print(f"loss on this minibatch (untrained, random init): {loss.item():.4f}")
print()


# ---------------------------------------------------------------
# Sanity check: random init gives loss ~ -log(1/V)
# ---------------------------------------------------------------
# A model with no information should assign roughly uniform probability
# to every vocab token. The expected loss is -log(1/V) = log(V).
import math
print(f"Expected loss for uniform output: log({V}) = {math.log(V):.4f}")
print()
# If our actual loss is much higher than this, the random init is WAY
# overconfident -- which is what happens with the default torch.randn
# initialization. We multiplied W1 and W2 by 0.1 above to prevent that.
# Lesson 04 dives deep into why initialization matters.


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - The full forward pass is 5 lines: lookup, flatten, linear+tanh,
#   linear, cross_entropy.
# - F.cross_entropy(logits, targets) handles softmax + NLL together,
#   numerically stable and faster than doing it manually.
# - At init, loss should be near log(V); a much higher value means
#   the initialization is bad.
#
# Next: train this thing.
