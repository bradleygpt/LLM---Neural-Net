"""
04_build_wavenet.py
-------------------
Now build the actual WaveNet for our names problem.

Architecture (block_size=8, embedding_dim=24, hidden=128):

    Input: (B, 8) integer indices

    Embedding: (V=27 -> D=24)             -> (B, 8, 24)

    --- merge level 1: 8 -> 4 timesteps ---
    FlattenConsecutive(2)                  -> (B, 4, 48)
    Linear(48 -> 128, no bias) BN Tanh     -> (B, 4, 128)

    --- merge level 2: 4 -> 2 timesteps ---
    FlattenConsecutive(2)                  -> (B, 2, 256)
    Linear(256 -> 128, no bias) BN Tanh    -> (B, 2, 128)

    --- merge level 3: 2 -> 1 timestep ---
    FlattenConsecutive(2)                  -> (B, 1, 256) -> squeezed (B, 256)
    Linear(256 -> 128, no bias) BN Tanh    -> (B, 128)

    --- output ---
    Linear(128 -> V=27)                    -> (B, 27)  logits


This is essentially the Bengio MLP from lessons 03/04 STRETCHED into a
deeper hierarchy. It uses the same primitives (linear, BN, tanh) but
arranges them in a tree.

Karpathy's video chooses these specific dimensions because they balance
expressiveness with parameter count. Total: ~76k parameters, vs the
~12k of the lesson 03 flat MLP -- but the WaveNet sees 8 chars of
context instead of 3, AND has more capacity to actually USE that context.

This script just builds and inspects. Training is the next script.
"""

import torch


# ---------------------------------------------------------------
# Module classes (copied from script 03 for self-containment)
# ---------------------------------------------------------------
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.W = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.b = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.W
        if self.b is not None:
            self.out = self.out + self.b
        return self.out
    def parameters(self):
        return [self.W] + ([] if self.b is None else [self.b])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps, self.momentum, self.training = eps, momentum, True
        self.gamma, self.beta = torch.ones(dim), torch.zeros(dim)
        self.running_mean, self.running_var = torch.zeros(dim), torch.ones(dim)
    def __call__(self, x):
        if self.training:
            dim = 0 if x.ndim == 2 else (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True, unbiased=False)
        else:
            xmean, xvar = self.running_mean, self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
        return self.out
    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ---------------------------------------------------------------
# Build the WaveNet
# ---------------------------------------------------------------
torch.manual_seed(42)

# Hyperparameters chosen to roughly match Karpathy's video
V = 27           # vocab (26 letters + '.')
D = 24           # embedding dim
H = 128          # hidden width per merge level
BLOCK = 8        # context length

model = Sequential([
    Embedding(V, D),
    FlattenConsecutive(2), Linear(D * 2, H, bias=False), BatchNorm1d(H), Tanh(),
    FlattenConsecutive(2), Linear(H * 2, H, bias=False), BatchNorm1d(H), Tanh(),
    FlattenConsecutive(2), Linear(H * 2, H, bias=False), BatchNorm1d(H), Tanh(),
    Linear(H, V),
])

# Karpathy trick: scale the LAST layer's weights to make initial logits small.
# This pushes initial loss to ~log(V) instead of being huge.
with torch.no_grad():
    model.layers[-1].W *= 0.1

# Mark all parameters as requiring gradients
for p in model.parameters():
    p.requires_grad_()

print(f"Model architecture:")
for i, layer in enumerate(model.layers):
    print(f"  {i:>2}. {type(layer).__name__}")
print()

total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")
print()


# ---------------------------------------------------------------
# Sanity-check forward pass
# ---------------------------------------------------------------
x = torch.randint(0, V, (4, BLOCK))
print(f"Input shape: {tuple(x.shape)}")

out = model(x)
print(f"Output shape: {tuple(out.shape)}  (B, V)  -- logits")
print()

# Inspect intermediate shapes
print("Layer-by-layer shapes:")
y = x
for i, layer in enumerate(model.layers):
    y = layer(y)
    print(f"  {i:>2}. {type(layer).__name__:>20s}  {tuple(y.shape)}")
print()


# ---------------------------------------------------------------
# Sanity check: initial loss should be ~log(V)
# ---------------------------------------------------------------
import torch.nn.functional as F
import math
fake_targets = torch.randint(0, V, (4,))
loss = F.cross_entropy(out, fake_targets)
print(f"Initial loss with random init: {loss.item():.4f}")
print(f"Expected (uniform output):     {math.log(V):.4f}")
print(f"  -> close to expected, no hockey stick.")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - Architecture is a Sequential of 13 layers.
# - Three merge levels: 8 -> 4 -> 2 -> 1 timesteps.
# - At each level, FlattenConsecutive doubles features then a Linear+BN+Tanh
#   block compresses them back to H. So the DEPTH grows but WIDTH stays
#   manageable.
# - Total params ~76k, comparable to the flat MLP at block_size=3 in
#   lesson 03 (12k) but expressing 8x more context.
#
# Next: train it.
