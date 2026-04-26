"""
01_why_flat_mlp_hits_a_wall.py
------------------------------
The MLP we built in lessons 03-05 takes a context of N characters,
flattens them into one big vector, and feeds that through a single
hidden layer. With block_size=3 and D=10, that flattened vector has
30 elements. Fine.

What if we want a longer context, say 8 characters? The flattened input
becomes 80 elements. The first hidden layer's weight matrix W1 has
shape (80, hidden_size). For hidden=200, that's 16,000 parameters in
W1 alone -- and ALL the context information has to fit through that
ONE big matmul.

The bigger problem: the network sees the entire context all at once,
in one undifferentiated soup of numbers. It can't tell what "position"
each character is in unless it learns to. Information at position 0
and position 7 hit the SAME hidden layer simultaneously, with no
hierarchy or structure.

WaveNet's idea: instead of mashing all 8 characters into one big vector
in one shot, MERGE them progressively in pairs.

    Layer 0:  [c0 c1 c2 c3 c4 c5 c6 c7]
                |   |    |   |    |   |    |   |
                v   v    v   v    v   v    v   v
                e0  e1   e2  e3   e4  e5   e6  e7    (embed each char)
                 \\ /     \\ /     \\ /     \\ /
    Layer 1:    h01      h23      h45      h67       (merge pairs)
                  \\        /         \\        /
    Layer 2:      h0123              h4567           (merge again)
                          \\        /
    Layer 3:               h01234567                 (final merge)
                              |
                              v
                            logits

At each level, the network combines TWO neighboring chunks. After
log2(block_size) = 3 levels, the context has been compressed into
a single vector representing all 8 characters with HIERARCHICAL
structure -- each layer can specialize in different combinations.

This script just sets up the dataset for WaveNet (block_size=8).
The next scripts build the architecture.
"""

import math
import random
import torch
from data_utils import load_names, build_vocab, build_dataset


# ---------------------------------------------------------------
# Build dataset with longer context
# ---------------------------------------------------------------
random.seed(42)
torch.manual_seed(42)

names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)
random.shuffle(names)
n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))
BLOCK = 8

Xtr, Ytr = build_dataset(names[:n1], stoi, block_size=BLOCK)
Xdev, Ydev = build_dataset(names[n1:n2], stoi, block_size=BLOCK)
Xte, Yte = build_dataset(names[n2:], stoi, block_size=BLOCK)

print(f"vocab size: {V}")
print(f"block_size: {BLOCK}")
print(f"train: {Xtr.shape}, dev: {Xdev.shape}, test: {Xte.shape}")
print()


# ---------------------------------------------------------------
# Show what an example looks like
# ---------------------------------------------------------------
print("First 5 training examples (block_size=8):")
for i in range(5):
    ctx = "".join(itos[c.item()] for c in Xtr[i])
    target = itos[Ytr[i].item()]
    print(f"  '{ctx}' --> '{target}'")
print()


# ---------------------------------------------------------------
# Compare flat MLP parameter cost at block_size=3 vs 8
# ---------------------------------------------------------------
D = 10
HIDDEN = 200

print(f"Flat MLP first-layer parameters with embedding D={D}, hidden={HIDDEN}:")
for b in [3, 5, 8, 16]:
    flat_size = b * D
    w1_params = flat_size * HIDDEN
    print(f"  block_size {b:>2}: flat input {flat_size:>4} dims, W1 = {w1_params:>7,} params")
print()

print(f"With WaveNet's hierarchical approach, the parameters DON'T scale")
print(f"linearly with context length. Each merge layer handles 2 inputs ->")
print(f"1 output, so the cost is constant per layer regardless of how long")
print(f"the context is. log2(block_size) layers covers it.")
print()


# ---------------------------------------------------------------
# Save the dataset for later scripts
# ---------------------------------------------------------------
torch.save({
    "Xtr": Xtr, "Ytr": Ytr, "Xdev": Xdev, "Ydev": Ydev, "Xte": Xte, "Yte": Yte,
    "stoi": stoi, "itos": itos, "block_size": BLOCK, "V": V,
}, "dataset.pt")
print("Saved dataset.pt for the next scripts.")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - Flat MLPs treat the context as one big vector. Information at
#   different positions has no structural distinction.
# - As context length grows, the first-layer parameters grow linearly,
#   but the network's ABILITY to use that context efficiently does not.
# - WaveNet's hierarchical merge structure handles long context with
#   log(N) layers and constant parameters per layer.
# - This is the same idea as a binary tree: O(log N) depth instead of
#   O(N) width.
#
# Next script: build the merge operation and start composing layers.
