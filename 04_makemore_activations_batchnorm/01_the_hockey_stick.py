"""
01_the_hockey_stick.py
----------------------
The previous lesson ended with a teaser: try running the MLP with PyTorch's
default `torch.randn` initialization (no `* 0.1`). The initial loss is
catastrophic, and the model spends thousands of steps just CLEANING UP
its own bad init before it can start learning.

This script demonstrates exactly that. We train two models that are
identical EXCEPT for initialization, and compare their loss curves.

The two key facts:

  1. With default init, the OUTPUT logits are huge random numbers. The
     softmax over them gives near-1 probability to ONE random class and
     near-0 to all others. That makes the cross-entropy loss enormous.

  2. The expected initial loss for a random classifier is `log(V)`. For
     V=27, that's about 3.3. ANY initial loss much above this is a
     dragon you have to slay before real training can start.

The fix preview:
    - Initialize the OUTPUT layer (W2, b2) very small or zero. This
      makes initial logits near zero, so all classes have ~uniform
      probability, so the initial loss is ~log(V).
    - Initialize hidden layers carefully too (Kaiming init, script 03).
"""

import torch
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

from data_utils import load_names, build_vocab, build_dataset


# ---------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------
names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)
random.seed(42)
random.shuffle(names)
n1 = int(0.8 * len(names))
Xtr, Ytr = build_dataset(names[:n1], stoi, block_size=3)
BLOCK = 3
print(f"Training set: {Xtr.shape[0]:,} examples, vocab size {V}")
print(f"Expected loss for uniform-output model: log({V}) = {math.log(V):.4f}")
print()


# ---------------------------------------------------------------
# Run a short training, return per-step losses
# ---------------------------------------------------------------
def train_short(init_scale_W1, init_scale_W2, steps=2000, lr=0.1, seed=42):
    torch.manual_seed(seed)
    D, H = 10, 200
    C = torch.randn((V, D))
    W1 = torch.randn((BLOCK * D, H)) * init_scale_W1
    b1 = torch.randn(H) * init_scale_W1
    W2 = torch.randn((H, V)) * init_scale_W2
    b2 = torch.randn(V) * init_scale_W2
    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad_()

    losses = []
    for step in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (32,))
        Xb, Yb = Xtr[ix], Ytr[ix]
        emb = C[Xb]
        flat = emb.view(emb.shape[0], -1)
        h = torch.tanh(flat @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)
        for p in params:
            p.grad = None
        loss.backward()
        for p in params:
            p.data += -lr * p.grad
        losses.append(loss.item())
    return losses


print("Run 1: BAD init (PyTorch defaults)")
losses_bad = train_short(init_scale_W1=1.0, init_scale_W2=1.0)
print(f"  step 0    loss: {losses_bad[0]:.2f}")
print(f"  step 100  loss: {losses_bad[100]:.2f}")
print(f"  step 500  loss: {losses_bad[500]:.2f}")
print(f"  step 1999 loss: {losses_bad[-1]:.2f}")
print()

print("Run 2: tiny W2/b2 (so initial logits ~ 0, near-uniform probs)")
losses_good = train_short(init_scale_W1=0.2, init_scale_W2=0.01)
print(f"  step 0    loss: {losses_good[0]:.2f}")
print(f"  step 100  loss: {losses_good[100]:.2f}")
print(f"  step 500  loss: {losses_good[500]:.2f}")
print(f"  step 1999 loss: {losses_good[-1]:.2f}")
print()


# ---------------------------------------------------------------
# Plot both
# ---------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(losses_bad, label=f"bad init (initial loss {losses_bad[0]:.1f})", alpha=0.7)
plt.plot(losses_good, label=f"good init (initial loss {losses_good[0]:.2f})", alpha=0.7)
plt.axhline(math.log(V), color="red", linestyle="--", label=f"log(V)={math.log(V):.2f}")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("The hockey stick: bad init wastes the first ~500 steps")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("hockey_stick.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved hockey_stick.png")
print()


# ---------------------------------------------------------------
# Why does the bad-init loss drop so fast at first?
# ---------------------------------------------------------------
# At step 0, the model is overconfident in random classes -- but training
# rapidly SHRINKS the output logits toward zero, which alone reduces loss
# from ~27 to ~3.3. That cliff is just the model UN-DOING its bad init.
# It's not learning anything about the data yet.
#
# The mechanism: gradient descent on cross-entropy naturally pushes
# logits toward small values when they're wrong (and they're ALL wrong
# at random init). So the first few hundred steps are just "stop being
# so confident", not "learn the patterns". A good init skips this.
print("KEY INSIGHT: a good init means more of your training steps are")
print("spent LEARNING, not RECOVERING.")
