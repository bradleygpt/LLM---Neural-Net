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
    ...

This is verbose. That's the point. Each named intermediate is a place we
can attach a manual gradient and verify it against autograd.

This script does the forward pass, lets autograd compute the reference
gradients, and saves everything (values + grads) for the next scripts.
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
HIDDEN = 64

N = 32
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
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)

hprebn = embcat @ W1 + b1

bnmeani = (1.0 / N) * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = (1.0 / (N - 1)) * bndiff2.sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

h = torch.tanh(hpreact)

logits = h @ W2 + b2

logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum ** -1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(N), Yb].mean()

print(f"forward pass complete. loss = {loss.item():.4f}")
print()


# ---------------------------------------------------------------
# Get autograd's gradient for everything (reference truth)
# ---------------------------------------------------------------
for p in parameters:
    p.grad = None

intermediates = {
    "logprobs": logprobs, "probs": probs, "counts": counts,
    "counts_sum": counts_sum, "counts_sum_inv": counts_sum_inv,
    "norm_logits": norm_logits, "logit_maxes": logit_maxes, "logits": logits,
    "h": h, "hpreact": hpreact, "bnraw": bnraw, "bnvar_inv": bnvar_inv,
    "bnvar": bnvar, "bndiff2": bndiff2, "bndiff": bndiff,
    "bnmeani": bnmeani, "hprebn": hprebn, "embcat": embcat, "emb": emb,
}
for t in intermediates.values():
    t.retain_grad()

loss.backward()

print("Reference (autograd) gradients computed for all named intermediates.")
print(f"Sample: logits.grad shape = {tuple(logits.grad.shape)}")
print(f"        max abs value = {logits.grad.abs().max().item():.6f}")
print()


# ---------------------------------------------------------------
# Save state for the next scripts
# ---------------------------------------------------------------
# CRITICAL detail: torch.save serializes tensor DATA but NOT .grad
# attributes. So we save grads under separate keys (e.g. "logprobs_grad")
# and reload them in the next scripts. The cmp() helper there uses the
# saved grad tensors as ground truth.
state = {
    "Xb": Xb, "Yb": Yb, "N": N, "V": V, "HIDDEN": HIDDEN, "BLOCK": BLOCK, "D": D,
    # parameter values (detached, since we don't need their grad attrs going forward)
    "C": C.detach(), "W1": W1.detach(), "b1": b1.detach(),
    "W2": W2.detach(), "b2": b2.detach(),
    "bngain": bngain.detach(), "bnbias": bnbias.detach(),
    # autograd's reference grads for parameters
    "C_grad": C.grad.clone(), "W1_grad": W1.grad.clone(), "b1_grad": b1.grad.clone(),
    "W2_grad": W2.grad.clone(), "b2_grad": b2.grad.clone(),
    "bngain_grad": bngain.grad.clone(), "bnbias_grad": bnbias.grad.clone(),
    "loss": loss.detach(),
}

# Add intermediate values + their reference grads from autograd
for name, t in intermediates.items():
    state[name] = t.detach()
    state[name + "_grad"] = t.grad.clone()

torch.save(state, "state.pt")
print("Saved forward state to state.pt (values + autograd reference gradients).")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - The forward pass has ~20 named intermediate tensors. Each has a
#   corresponding gradient dL/d(intermediate) that autograd computed.
# - We saved BOTH values and autograd's gradients to state.pt.
# - One PyTorch gotcha: torch.save doesn't preserve .grad attributes,
#   so we explicitly save grads under "<name>_grad" keys.
#
# Our job in the next scripts: compute each gradient BY HAND and verify
# it matches the saved reference to within numerical precision.
