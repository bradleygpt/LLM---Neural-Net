"""
06_final_mlp_with_batchnorm.py
------------------------------
The final synthesis. Build a deep MLP using everything from this lesson:

    - Embedding lookup
    - Multiple hidden layers
    - Kaiming-style init
    - BatchNorm after each linear layer
    - Tanh nonlinearities
    - Cross-entropy loss
    - LR-decayed minibatch SGD

We'll structure it like real PyTorch code: each "Linear", "BatchNorm",
and "Tanh" is a Module-like class with a __call__ and a parameters()
method. This anticipates the structure of nn.Sequential and points
directly toward the transformer code in lesson 07.
"""

import torch
import torch.nn.functional as F
import random

from data_utils import load_names, build_vocab, build_dataset


# ---------------------------------------------------------------
# Module-like classes
# ---------------------------------------------------------------

class Linear:
    def __init__(self, fan_in, fan_out, bias=True, gain=1.0):
        self.W = torch.randn(fan_in, fan_out) * gain / fan_in**0.5
        self.b = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out
    def parameters(self):
        return [self.W] + ([] if self.b is None else [self.b])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps; self.momentum = momentum; self.training = True
        self.gamma = torch.ones(dim); self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim); self.running_var = torch.ones(dim)
    def __call__(self, x):
        if self.training:
            mean = x.mean(0, keepdim=True); var = x.var(0, keepdim=True, unbiased=False)
        else:
            mean = self.running_mean; var = self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze(0)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze(0)
        return out
    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x): return torch.tanh(x)
    def parameters(self): return []


# ---------------------------------------------------------------
# Build the network
# ---------------------------------------------------------------
random.seed(42)
torch.manual_seed(42)

names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)
random.shuffle(names)
n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))
Xtr, Ytr = build_dataset(names[:n1], stoi, block_size=3)
Xdev, Ydev = build_dataset(names[n1:n2], stoi, block_size=3)
BLOCK = 3

D, H = 10, 100
C = torch.randn(V, D)

# A "Sequential"-style stack
layers = [
    Linear(BLOCK * D, H, gain=5/3), BatchNorm1d(H), Tanh(),
    Linear(H, H, gain=5/3),         BatchNorm1d(H), Tanh(),
    Linear(H, H, gain=5/3),         BatchNorm1d(H), Tanh(),
    Linear(H, H, gain=5/3),         BatchNorm1d(H), Tanh(),
    Linear(H, V, gain=1.0),
]
# Karpathy trick: scale the last layer down further so initial logits ~ 0
with torch.no_grad():
    layers[-1].W *= 0.1

parameters = [C] + [p for l in layers for p in l.parameters()]
for p in parameters:
    p.requires_grad_()
print(f"Total params: {sum(p.numel() for p in parameters):,}")
print()


# ---------------------------------------------------------------
# Forward function
# ---------------------------------------------------------------
def forward(Xb):
    emb = C[Xb]
    h = emb.view(emb.shape[0], -1)
    for layer in layers:
        h = layer(h)
    return h


# ---------------------------------------------------------------
# Train
# ---------------------------------------------------------------
STEPS = 50_000          # bump up for serious training
BATCH = 32

for step in range(STEPS):
    ix = torch.randint(0, Xtr.shape[0], (BATCH,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = forward(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if step < 0.7 * STEPS else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if step % 5000 == 0:
        print(f"step {step:>6}/{STEPS}  loss = {loss.item():.4f}")
print()


# ---------------------------------------------------------------
# Evaluate (eval mode for BN)
# ---------------------------------------------------------------
@torch.no_grad()
def split_loss(X, Y):
    # Switch all BN layers to eval mode
    for layer in layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = False
    logits = forward(X)
    loss = F.cross_entropy(logits, Y).item()
    # restore
    for layer in layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = True
    return loss


print(f"FINAL train loss: {split_loss(Xtr, Ytr):.4f}")
print(f"FINAL dev   loss: {split_loss(Xdev, Ydev):.4f}")
print()


# ---------------------------------------------------------------
# Sample
# ---------------------------------------------------------------
@torch.no_grad()
def sample(g):
    for layer in layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = False
    out = []
    context = [0] * BLOCK
    while True:
        x = torch.tensor([context])
        logits = forward(x)
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
        if len(out) > 50: break
    return "".join(out)


g = torch.Generator().manual_seed(2147483647 + 10)
print("20 names from the deep MLP with BatchNorm:")
for _ in range(20):
    print(f"  {sample(g)}")
print()


# ---------------------------------------------------------------
# What we just learned (and the path forward)
# ---------------------------------------------------------------
print("""
This is the recipe that works for any feedforward net up to GPT scale:

  - Embedding lookup at the input
  - Stack of (Linear, Norm, Activation) blocks
  - Linear output head
  - Cross-entropy loss
  - Minibatch SGD with LR decay
  - Diagnostic plots when things go wrong

The next videos in the series do three things:

  1. (Lesson 05) Backprop everything by hand instead of using
     loss.backward(). The mechanics get burned into your brain.

  2. (Lesson 06) Replace the flat MLP with a hierarchical "WaveNet"
     architecture so longer contexts compose information progressively.

  3. (Lesson 07) Replace the entire stack with a TRANSFORMER, where
     attention replaces fixed-window flattening. You get GPT.

Everything you've built so far is a stepping stone to that transformer.
The Linear/BatchNorm/Tanh class structure here is *exactly* how the
transformer code in lesson 07 is organized -- different blocks, same
shape.
""")
