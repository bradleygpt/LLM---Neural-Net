"""
03_embeddings.py
----------------
The single most important new idea in this lesson: EMBEDDINGS.

Each character in our vocabulary gets a learned vector of dimension D
(say D=2 for visualization, D=10 for real training). These vectors live
in a single big lookup table C of shape (V, D).

To get the embedding of character index i:  C[i].
To get the embeddings for a whole context [i1, i2, i3]:  C[[i1, i2, i3]].

Pytorch makes this beautifully clean -- you just INDEX a tensor with another
tensor, and the result has shape (..., D). For a batch of contexts of
shape (N, block_size), the result has shape (N, block_size, D).

Why is this powerful? Two reasons:

  1. Characters that BEHAVE similarly (e.g. vowels, or consonants that
     often appear in the same positions) get pulled toward similar
     vectors during training. The model learns its own "linguistic
     features" in continuous space. We didn't tell it about vowels.

  2. Two contexts that are SIMILAR but not identical (e.g. "...mar" and
     "...mer") have similar embeddings, so the network's output for one
     is similar to its output for the other. INFORMATION IS SHARED across
     contexts. Counting models can't do this -- each cell is independent.

This script stops short of training -- we just build C, index into it,
and visualize the embeddings of an untrained model so you can see the
shape of the operation.
"""

import torch
import matplotlib
matplotlib.use("Agg")  # headless: write to file instead of opening a window
import matplotlib.pyplot as plt


# Load the dataset
data = torch.load("dataset.pt", weights_only=False)
Xtr, Ytr = data["Xtr"], data["Ytr"]
itos = data["itos"]
V = len(itos)


# ---------------------------------------------------------------
# Build the embedding table C
# ---------------------------------------------------------------
# Shape: (V, D). One row per vocab character. Initialized randomly.
# D=2 is for visualization. Real training uses D=10 or so.
torch.manual_seed(2147483647)
D = 2
C = torch.randn((V, D))
print(f"Embedding table C has shape {C.shape}")
print(f"  C[0] (the '.' boundary token) = {C[0]}")
print(f"  C[1] (whatever 'a' is)        = {C[1]}")
print()


# ---------------------------------------------------------------
# The lookup is just indexing
# ---------------------------------------------------------------
# C[5] gives a (D,) vector. C[[5, 12, 7]] gives a (3, D) tensor.
# C[Xtr] -- where Xtr has shape (N, block_size) -- gives (N, block_size, D).

emb = C[Xtr]
print(f"Xtr.shape  = {tuple(Xtr.shape)}  (N rows, each with block_size context indices)")
print(f"emb.shape  = {tuple(emb.shape)}  (N, block_size, D)")
print()


# ---------------------------------------------------------------
# Compare with one-hot @ C (they're equivalent)
# ---------------------------------------------------------------
# In script 06 of the bigram lesson we did x_onehot @ W. That's identical
# to C[x] when W=C. Indexing is just a fast version of one-hot matmul.
import torch.nn.functional as F
oh = F.one_hot(Xtr[:5], num_classes=V).float()  # (5, block_size, V)
via_onehot = oh @ C                             # (5, block_size, D)
via_index = C[Xtr[:5]]                          # (5, block_size, D)
print("Two ways to do the embedding lookup -- they agree:")
print(f"  max abs difference: {(via_onehot - via_index).abs().max().item():.2e}")
print()


# ---------------------------------------------------------------
# Visualize the (random, untrained) embeddings in 2D
# ---------------------------------------------------------------
# We'll regenerate this plot AFTER training so you can see what the
# model learned (script 06).
plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0].detach(), C[:, 1].detach(), s=200)
for i in range(V):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
             ha="center", va="center", color="white", fontweight="bold")
plt.title("Random (untrained) character embeddings")
plt.grid(alpha=0.3)
plt.savefig("embeddings_untrained.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved embeddings_untrained.png -- random scatter. Boring.")
print("Compare to the post-training plot from script 06.")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - C is a (V, D) lookup table of learned vectors.
# - Indexing C with a tensor of integer indices gives a tensor of vectors.
# - This replaces one-hot encoding with a much smaller, denser representation.
# - C is just parameters -- it gets gradients in backprop, just like any
#   other weight matrix. The "lookup" operation has a trivial gradient:
#   the gradient flowing in goes straight into the rows of C that were
#   indexed. No-op for the rest.
#
# Next: build the full Bengio MLP architecture around this lookup.
