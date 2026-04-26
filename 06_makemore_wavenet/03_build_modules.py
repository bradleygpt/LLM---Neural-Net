"""
03_build_modules.py
-------------------
Set up the building blocks. We're following PyTorch's nn.Module style
but keeping everything simple for visibility:

    - Linear(fan_in, fan_out, bias)
    - BatchNorm1d(dim)         -- with the 3D-aware fix from Karpathy's video
    - Tanh
    - Embedding(num_emb, emb_dim)
    - FlattenConsecutive(n)
    - Sequential(*layers)

Each module exposes:
    .__call__(x)        -- forward pass
    .parameters()       -- list of learnable tensors
    .out                -- last output (saved for inspection during training)

The trickiest one is BatchNorm1d. Lesson 04's version assumed (B, C)
input. Here we pass (B, T, C) BEFORE the FlattenConsecutive collapses
the time dim away. So BN needs to handle EITHER shape -- normalize over
the BATCH dim if the input is 2D, or over BATCH+TIME if it's 3D.

PyTorch's actual nn.BatchNorm1d does a slightly different thing (it
normalizes per channel across batch with input as (B, C) or (B, C, L)).
We follow Karpathy's pragmatic choice from the video: just normalize
over all the non-feature dims. Makes the code symmetric and works.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------
# Linear: y = x @ W + b
# ---------------------------------------------------------------
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        # Kaiming-style init for tanh nonlinearity
        self.W = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.b = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.W
        if self.b is not None:
            self.out = self.out + self.b
        return self.out

    def parameters(self):
        return [self.W] + ([] if self.b is None else [self.b])


# ---------------------------------------------------------------
# BatchNorm1d: handles both (B, C) and (B, T, C) inputs
# ---------------------------------------------------------------
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # learnable params
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # running stats for inference
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            # Decide which dims to reduce over.
            # 2D input (B, C):       reduce over batch dim 0
            # 3D input (B, T, C):    reduce over (B, T) -> dims (0, 1)
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True, unbiased=False)  # use biased=False to avoid degenerate batches
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# ---------------------------------------------------------------
# Tanh: simple wrapper
# ---------------------------------------------------------------
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


# ---------------------------------------------------------------
# Embedding: lookup table
# ---------------------------------------------------------------
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


# ---------------------------------------------------------------
# FlattenConsecutive: reshape (B, T, C) -> (B, T/n, C*n)
# ---------------------------------------------------------------
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


# ---------------------------------------------------------------
# Sequential: chain modules together
# ---------------------------------------------------------------
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
# Test: build a tiny network and run a forward pass
# ---------------------------------------------------------------
torch.manual_seed(42)

V = 27
D = 10
H = 32
BLOCK = 8

model = Sequential([
    Embedding(V, D),
    FlattenConsecutive(2), Linear(D * 2, H, bias=False), BatchNorm1d(H), Tanh(),
    FlattenConsecutive(2), Linear(H * 2, H, bias=False), BatchNorm1d(H), Tanh(),
    FlattenConsecutive(2), Linear(H * 2, H, bias=False), BatchNorm1d(H), Tanh(),
    Linear(H, V),
])

# Mark all as needing gradients
for p in model.parameters():
    p.requires_grad_()

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
print()


# Try a forward pass on a small batch
B = 4
x = torch.randint(0, V, (B, BLOCK))
print(f"Input shape: {tuple(x.shape)}")

out = model(x)
print(f"Output shape: {tuple(out.shape)}")
print()


# Inspect intermediate shapes layer by layer
print("Intermediate shapes:")
y = x
for i, layer in enumerate(model.layers):
    y = layer(y)
    layer_name = type(layer).__name__
    print(f"  layer {i:>2} ({layer_name:>20}): {tuple(y.shape)}")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - We have a clean module-style API: Linear, BatchNorm1d, Tanh,
#   Embedding, FlattenConsecutive, Sequential.
# - BatchNorm needs to handle both 2D (B, C) and 3D (B, T, C) inputs.
# - The model is just a Sequential of 13 layers, defined in 1 expression.
# - Watch the shape progression: 8 timesteps -> 4 -> 2 -> 1, while
#   features get richer at each level.
#
# Next: train the WaveNet.
