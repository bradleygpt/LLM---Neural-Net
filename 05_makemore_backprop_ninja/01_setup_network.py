"""
01_setup_network.py
-------------------
Before we backprop anything by hand, we need a forward pass where every
intermediate value has a name. In normal PyTorch you'd write:

    h = torch.tanh(emb.view(N, -1) @ W1 + b1)

That's clean but useless for our purposes -- we can't compute the gradient
of `loss` with respect to `emb @ W1` if we never gave that intermediate
a name.

So we rewrite the forward pass with EVERY intermediate exposed:

    embcat = emb.view(N, -1)
    hprebn = embcat @ W1 + b1
    bnmeani = (1/N) * hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff ** 2
    bnvar = (1/(N-1)) * bndiff2.sum(0, keepdim=True)
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    # ... and so on for cross-entropy

This is verbose. That's the point. Each named intermediate is a place we
can attach a manual gradient and verify it against autograd.

This script just builds the network and shows the forward pass with all
intermediates exposed. Future scripts will add manual gradients one piece
at a time.
"""

import random
import torch
import torch.nn.functional as F
from data_utils import load_names, build_vocab, build_dataset


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
random.seed(42)
torch.manual_seed(42)

names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)
random.shuffle(names)
n1 = int(0.8 * len(names))
Xtr, Ytr = build_dataset(names[:n1], stoi, block_size=3)
print(f"Training set: {Xtr.shape[0]:,} examples, vocab size {V}")
print()


# ---------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------
BLOCK = 3
D = 10
HIDDEN = 64    # smaller than lesson 04 -- keeps the manual grad math tractable

# Take a smallish minibatch to make manual checks fast
N = 32           # batch size
ix = torch.randint(0, Xtr.shape[0], (N,))
Xb, Yb = Xtr[ix], Ytr[ix]


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((V, D),                     generator=g)
W1 = torch.randn((BLOCK * D, HIDDEN),       generator=g) * (5/3) / ((BLOCK * D) ** 0.5)
b1 = torch.randn(HIDDEN,                    generator=g) * 0.1
W2 = torch.randn((HIDDEN, V),               generator=g) * 0.1
b2 = torch.randn(V,                         generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, HIDDEN), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, HIDDEN), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad_()
print(f"Total params: {sum(p.numel() for p in parameters)}")
print()


# ---------------------------------------------------------------
# FORWARD PASS, WITH EVERY INTERMEDIATE NAMED
# ---------------------------------------------------------------
# Layer 1: embedding lookup
emb = C[Xb]                     # (N, BLOCK, D)
embcat = emb.view(emb.shape[0], -1)  # (N, BLOCK * D)

# Linear 1
hprebn = embcat @ W1 + b1       # (N, HIDDEN)  -- "pre-batchnorm hidden"

# BatchNorm: forward, broken into atomic steps
# We write out every step (mean, diff, var, etc.) so we can backprop each.
bnmeani = (1.0 / N) * hprebn.sum(0, keepdim=True)        # (1, HIDDEN)
bndiff = hprebn - bnmeani                                 # (N, HIDDEN)
bndiff2 = bndiff ** 2                                     # (N, HIDDEN)
bnvar = (1.0 / (N - 1)) * bndiff2.sum(0, keepdim=True)    # (1, HIDDEN), Bessel-corrected
bnvar_inv = (bnvar + 1e-5) ** -0.5                        # (1, HIDDEN)
bnraw = bndiff * bnvar_inv                                # (N, HIDDEN)
hpreact = bngain * bnraw + bnbias                         # (N, HIDDEN)

# Nonlinearity
h = torch.tanh(hpreact)                                   # (N, HIDDEN)

# Linear 2
logits = h @ W2 + b2                                      # (N, V)

# Cross-entropy loss, broken into pieces
# (we WON'T use F.cross_entropy because we want to backprop through softmax+NLL ourselves)
logit_maxes = logits.max(1, keepdim=True).values          # (N, 1) for numerical stability
norm_logits = logits - logit_maxes                        # (N, V)
counts = norm_logits.exp()                                # (N, V)
counts_sum = counts.sum(1, keepdim=True)                  # (N, 1)
counts_sum_inv = counts_sum ** -1                         # (N, 1)
probs = counts * counts_sum_inv                           # (N, V)
logprobs = probs.log()                                    # (N, V)
loss = -logprobs[range(N), Yb].mean()                     # scalar

print(f"forward pass complete. loss = {loss.item():.4f}")
print()


# ---------------------------------------------------------------
# Get autograd's gradient for everything (our reference truth)
# ---------------------------------------------------------------
for p in parameters:
    p.grad = None
# Also need gradients for the intermediates -- mark them as tensors we want grads on
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, norm_logits,
          logit_maxes, logits, h, hpreact, bnraw, bnvar_inv, bnvar, bndiff2,
          bndiff, bnmeani, hprebn, embcat, emb]:
    t.retain_grad()
loss.backward()

print("Reference (autograd) gradients computed for all named intermediates.")
print(f"Sample: logits.grad shape = {tuple(logits.grad.shape)}")
print(f"        max abs value = {logits.grad.abs().max().item():.6f}")
print()


# ---------------------------------------------------------------
# Save the state for the next script
# ---------------------------------------------------------------
state = {
    # input data
    "Xb": Xb, "Yb": Yb, "N": N, "V": V, "HIDDEN": HIDDEN, "BLOCK": BLOCK, "D": D,
    # parameters
    "C": C, "W1": W1, "b1": b1, "W2": W2, "b2": b2,
    "bngain": bngain, "bnbias": bnbias,
    # forward intermediates
    "emb": emb, "embcat": embcat, "hprebn": hprebn,
    "bnmeani": bnmeani, "bndiff": bndiff, "bndiff2": bndiff2,
    "bnvar": bnvar, "bnvar_inv": bnvar_inv, "bnraw": bnraw,
    "hpreact": hpreact, "h": h, "logits": logits,
    "logit_maxes": logit_maxes, "norm_logits": norm_logits,
    "counts": counts, "counts_sum": counts_sum, "counts_sum_inv": counts_sum_inv,
    "probs": probs, "logprobs": logprobs, "loss": loss,
}
torch.save(state, "state.pt")
print("Saved forward state to state.pt for use by the next scripts.")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - The forward pass has 20+ named intermediate tensors. Each one has
#   a corresponding gradient dL/d(intermediate) that autograd computed.
# - We've stored both the values and the autograd-computed gradients.
# - Our job in the next scripts: compute each gradient BY HAND and
#   verify it matches autograd to within numerical precision.
#
# The cmp() helper for verification:
#
#     def cmp(name, dt, t):
#         ex = torch.all(dt == t.grad).item()
#         app = torch.allclose(dt, t.grad)
#         maxdiff = (dt - t.grad).abs().max().item()
#         print(f"{name:>15}: exact={ex} | approx={app} | max diff = {maxdiff}")
#
# We want exact=True (or at minimum approx=True with maxdiff < 1e-7).
