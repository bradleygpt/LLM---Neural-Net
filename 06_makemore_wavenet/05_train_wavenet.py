"""
05_train_wavenet.py
-------------------
Train the WaveNet on the names dataset.

Same training loop pattern as lesson 04: minibatch SGD with LR decay,
loss tracking, eval on dev set. We use loss.backward() (you've already
graduated from manual backward last lesson; no need to redo it).

Expected outcome: final dev loss around 1.99-2.05 -- better than the
flat MLP's 2.10. The improvement comes from longer context (8 chars
vs 3) being processed hierarchically.
"""

import random
import torch
import torch.nn.functional as F
import math
from data_utils import load_names, build_vocab, build_dataset


# ---------------------------------------------------------------
# Module classes (copied for self-containment)
# ---------------------------------------------------------------
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.W = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.b = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.W
        if self.b is not None: self.out = self.out + self.b
        return self.out
    def parameters(self): return [self.W] + ([] if self.b is None else [self.b])


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
    def parameters(self): return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x); return self.out
    def parameters(self): return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    def __call__(self, IX):
        self.out = self.weight[IX]; return self.out
    def parameters(self): return [self.weight]


class FlattenConsecutive:
    def __init__(self, n): self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1: x = x.squeeze(1)
        self.out = x
        return self.out
    def parameters(self): return []


class Sequential:
    def __init__(self, layers): self.layers = layers
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        self.out = x
        return self.out
    def parameters(self): return [p for l in self.layers for p in l.parameters()]


# ---------------------------------------------------------------
# Setup
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

Xtr, Ytr = build_dataset(names[:n1], stoi, BLOCK)
Xdev, Ydev = build_dataset(names[n1:n2], stoi, BLOCK)
print(f"Training set: {Xtr.shape[0]:,} examples")


# ---------------------------------------------------------------
# Build the WaveNet
# ---------------------------------------------------------------
D, H = 24, 128

model = Sequential([
    Embedding(V, D),
    FlattenConsecutive(2), Linear(D * 2, H, bias=False), BatchNorm1d(H), Tanh(),
    FlattenConsecutive(2), Linear(H * 2, H, bias=False), BatchNorm1d(H), Tanh(),
    FlattenConsecutive(2), Linear(H * 2, H, bias=False), BatchNorm1d(H), Tanh(),
    Linear(H, V),
])

with torch.no_grad():
    model.layers[-1].W *= 0.1

for p in model.parameters():
    p.requires_grad_()

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print()


# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------
STEPS = 200_000     # higher than previous lessons -- WaveNet has more capacity
N_BATCH = 32

print(f"Training for {STEPS:,} steps...")
print()

losses = []
for step in range(STEPS):
    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (N_BATCH,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    # backward
    for p in model.parameters():
        p.grad = None
    loss.backward()

    # update with LR decay
    lr = 0.1 if step < 0.7 * STEPS else 0.01
    for p in model.parameters():
        p.data -= lr * p.grad

    losses.append(loss.log10().item())
    if step % 20_000 == 0:
        print(f"step {step:>7}/{STEPS}  loss = {loss.item():.4f}")

print(f"step {STEPS:>7}/{STEPS}  loss = {loss.item():.4f}")
print()


# ---------------------------------------------------------------
# Evaluate (switch BN to eval mode so it uses running stats)
# ---------------------------------------------------------------
@torch.no_grad()
def split_loss(X, Y):
    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = False
    logits = model(X)
    loss = F.cross_entropy(logits, Y).item()
    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = True
    return loss


print(f"FINAL train loss: {split_loss(Xtr, Ytr):.4f}")
print(f"FINAL dev   loss: {split_loss(Xdev, Ydev):.4f}")
print()


# ---------------------------------------------------------------
# Sample names from the trained WaveNet
# ---------------------------------------------------------------
@torch.no_grad()
def sample(g):
    # eval mode for sampling
    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = False

    out = []
    context = [0] * BLOCK
    while True:
        x = torch.tensor([context])
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
        if len(out) > 50:
            break

    # restore training mode
    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = True
    return "".join(out)


sg = torch.Generator().manual_seed(2147483647 + 10)
print("20 names from the trained WaveNet:")
for _ in range(20):
    print(f"  {sample(sg)}")
print()


# ---------------------------------------------------------------
# Save the trained model + the loss curve for the next script
# ---------------------------------------------------------------
state = {
    "params": [p.detach() for p in model.parameters()],
    "losses": losses,
    "stoi": stoi, "itos": itos,
    "block": BLOCK, "V": V, "D": D, "H": H,
}
torch.save(state, "trained_wavenet.pt")
print("Saved trained_wavenet.pt + loss history.")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - WaveNet trains the same way as the flat MLP (forward, loss, backward,
#   update). The architecture changed; the recipe didn't.
# - With block_size=8 and the hierarchical structure, dev loss should
#   land around 1.99-2.05.
# - That's a meaningful improvement over the flat MLP's ~2.10 with
#   block_size=3, achieved via LONGER CONTEXT not just more parameters.
# - Sample names will look noticeably more name-like, especially in the
#   late chars where 8-char context dominates over 3-char.
#
# Next: compare to flat MLP and the architecture's connection to
# convolutions / transformers.
