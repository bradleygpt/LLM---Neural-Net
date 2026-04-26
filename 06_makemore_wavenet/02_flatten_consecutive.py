"""
02_flatten_consecutive.py
-------------------------
The single new operation we need: FlattenConsecutive(n).

It takes a tensor of shape (batch, time, features) and groups every n
consecutive time steps into one expanded feature vector. Result shape:
(batch, time/n, features*n).

Concretely with n=2:
    Input  : (32, 8, 10)   -- 32 examples, 8 timesteps, 10 features each
    Output : (32, 4, 20)   -- 32 examples, 4 PAIRS, 20 features per pair

Why this matters: after FlattenConsecutive(2), a Linear layer with
output size H takes a 20-dim input and produces an H-dim output. That
H-dim output represents "merged information from the 2 input timesteps."
We can then call FlattenConsecutive(2) again on the result to merge
PAIRS OF MERGED PAIRS, producing one vector that summarizes 4 timesteps.
And again to summarize 8.

This is how a hierarchical tree gets built using nothing but reshape +
Linear. No new architecture primitives needed.

The implementation is just torch.view -- no compute, no parameters. But
the MEANING is profound: it's how the network learns hierarchical
structure.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------
# The FlattenConsecutive class
# ---------------------------------------------------------------
class FlattenConsecutive:
    """
    Reshape (B, T, C) -> (B, T // n, C * n) by grouping every n
    consecutive timesteps into one expanded feature vector.
    """
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        # If we end up with T=1 (the final merge), squeeze out that dim
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []


# ---------------------------------------------------------------
# Demo: how the shapes evolve
# ---------------------------------------------------------------
torch.manual_seed(42)

B, T, C = 4, 8, 3
x = torch.randn(B, T, C)
print(f"Input shape: {tuple(x.shape)}  (batch, time, features)")
print()

# Apply FlattenConsecutive(2) -- groups pairs
fc = FlattenConsecutive(2)
y = fc(x)
print(f"After FlattenConsecutive(2): {tuple(y.shape)}")
print(f"  -> 8 timesteps merged into 4 pairs, features doubled to {C*2}")
print()

# Apply again
y = fc(y)
print(f"After another FlattenConsecutive(2): {tuple(y.shape)}")
print(f"  -> 4 pairs merged into 2 quads, features doubled to {C*4}")
print()

# Once more
y = fc(y)
print(f"After a third FlattenConsecutive(2): {tuple(y.shape)}")
print(f"  -> 2 quads merged into 1 'octet', features = {C*8}")
print(f"  (squeezed the singleton time dim -- now (B, features))")
print()


# ---------------------------------------------------------------
# Why is this the same data, just reshaped?
# ---------------------------------------------------------------
# torch.view doesn't copy memory. It reinterprets the existing tensor's
# elements as a different shape. The original (B, T, C) tensor lays out
# B groups of T*C numbers each. View into (B, T/n, C*n) just chooses
# a different "grid" over the same numbers.
#
# Concretely: x[b, t1, c] and x[b, t2, c] for adjacent t1, t2 sit next
# to each other in memory, so when we reshape, those two C-vectors
# become a contiguous 2C vector. That's the "merge."

print("View vs reshape vs copy:")
x = torch.tensor([[[1., 2.], [3., 4.], [5., 6.], [7., 8.]]])  # (1, 4, 2)
print(f"  Original (1, 4, 2): {x.tolist()}")
y = x.view(1, 2, 4)
print(f"  After view(1, 2, 4): {y.tolist()}")
print(f"  Notice: each row of y is two consecutive (c0, c1) pairs from x.")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - FlattenConsecutive(n) is just .view(). Zero parameters, zero compute.
# - It transforms (B, T, C) -> (B, T/n, C*n) by grouping consecutive
#   time steps into one merged feature vector.
# - The "merging" is a memory layout reinterpretation, not a real op.
# - The actual learning happens in the Linear layer that comes AFTER
#   this reshape -- the linear layer sees C*n inputs and produces a
#   merged representation.
# - Stacking k of these (with Linears between) compresses 2^k timesteps
#   into one vector via a binary tree structure.
#
# Next: stack a few of these into a real WaveNet.
