"""
06_train_with_manual_grads.py
-----------------------------
The payoff. Train the network end-to-end using ONLY the manual gradients
we derived in scripts 02-05. No loss.backward(). No autograd.

This script integrates everything:
    - Forward pass with all intermediates exposed
    - Manual backward pass (the simplified BatchNorm form for speed)
    - Parameter updates with SGD
    - BatchNorm RUNNING STATS for inference (critical -- see below)
    - Loss tracking + sampling

A subtle but important detail: BatchNorm computes mean/std FROM THE BATCH
during training. At inference time, you typically only have one example,
so batch stats are degenerate (var with batch=1 has zero degrees of
freedom and produces NaN). The fix every real implementation uses:
maintain RUNNING ESTIMATES of mean and variance during training, and use
those frozen estimates at inference. We do that here.
"""

import random
import torch
import torch.nn.functional as F
from data_utils import load_names, build_vocab, build_dataset


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
Xtr, Ytr = build_dataset(names[:n1], stoi, block_size=3)
Xdev, Ydev = build_dataset(names[n1:n2], stoi, block_size=3)
print(f"Training set: {Xtr.shape[0]:,} examples")

BLOCK, D, HIDDEN = 3, 10, 64
N_BATCH = 32
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((V, D),                 generator=g)
W1 = torch.randn((BLOCK*D, HIDDEN),     generator=g) * (5/3) / ((BLOCK*D)**0.5)
b1 = torch.randn(HIDDEN,                generator=g) * 0.1
W2 = torch.randn((HIDDEN, V),           generator=g) * 0.1
b2 = torch.randn(V,                     generator=g) * 0.1
bngain = torch.randn((1, HIDDEN), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, HIDDEN), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(f"Total params: {sum(p.numel() for p in parameters):,}")
print()

# Running stats for BatchNorm at inference. Updated during training,
# used at inference time so we don't divide by zero on batch-of-1.
bnmean_running = torch.zeros((1, HIDDEN))
bnvar_running = torch.ones((1, HIDDEN))


# ---------------------------------------------------------------
# Training loop with manual backward
# ---------------------------------------------------------------
STEPS = 50_000
print(f"Training for {STEPS:,} steps using ONLY manual gradients...")
print()

for step in range(STEPS):
    # --- minibatch ---
    ix = torch.randint(0, Xtr.shape[0], (N_BATCH,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # --- forward (all intermediates exposed) ---
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1

    bnmean = hprebn.mean(0, keepdim=True)
    bnvar = hprebn.var(0, keepdim=True, unbiased=True)
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias

    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    # Update running stats with EMA (used at inference)
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmean
        bnvar_running = 0.999 * bnvar_running + 0.001 * bnvar

    # --- manual backward, end to end ---
    # 1. cross-entropy + softmax => standard closed form
    dlogits = F.softmax(logits, dim=1)
    dlogits[range(N_BATCH), Yb] -= 1
    dlogits /= N_BATCH

    # 2. linear 2:  logits = h @ W2 + b2
    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)

    # 3. tanh
    dhpreact = (1 - h**2) * dh

    # 4. BatchNorm (simplified single-line backward)
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)
    dhprebn = (bngain * bnvar_inv / N_BATCH) * (
        N_BATCH * dhpreact
        - dhpreact.sum(0)
        - (N_BATCH / (N_BATCH - 1)) * bnraw * (dhpreact * bnraw).sum(0)
    )

    # 5. linear 1
    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)

    # 6. reshape
    demb = dembcat.view(emb.shape)

    # 7. embedding lookup -- accumulate gradients into rows of C
    dC = torch.zeros_like(C)
    for i in range(N_BATCH):
        for j in range(BLOCK):
            dC[Xb[i, j]] += demb[i, j]

    # --- update ---
    grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
    lr = 0.1 if step < 0.7 * STEPS else 0.01
    for p, dp in zip(parameters, grads):
        p.data -= lr * dp

    if step % 5000 == 0:
        print(f"step {step:>6}/{STEPS}  loss = {loss.item():.4f}")

print()


# ---------------------------------------------------------------
# Evaluate using running stats (inference mode)
# ---------------------------------------------------------------
@torch.no_grad()
def split_loss(X, Y):
    emb = C[X]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1
    # Use running stats (frozen from training) instead of batch stats
    bnraw = (hprebn - bnmean_running) * (bnvar_running + 1e-5) ** -0.5
    hpreact = bngain * bnraw + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()


print(f"FINAL train loss: {split_loss(Xtr, Ytr):.4f}")
print(f"FINAL dev   loss: {split_loss(Xdev, Ydev):.4f}")
print()


# ---------------------------------------------------------------
# Sample
# ---------------------------------------------------------------
@torch.no_grad()
def sample(g):
    out = []
    context = [0] * BLOCK
    while True:
        x = torch.tensor([context])
        emb_s = C[x]
        embcat_s = emb_s.view(1, -1)
        hprebn_s = embcat_s @ W1 + b1
        # Use running stats -- avoids div-by-zero with batch size 1
        bnraw_s = (hprebn_s - bnmean_running) * (bnvar_running + 1e-5) ** -0.5
        h_s = torch.tanh(bngain * bnraw_s + bnbias)
        logits_s = h_s @ W2 + b2
        probs = F.softmax(logits_s, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
        if len(out) > 50: break
    return "".join(out)


sg = torch.Generator().manual_seed(2147483647 + 10)
print("20 names from the manually-trained network:")
for _ in range(20):
    print(f"  {sample(sg)}")
print()


# ---------------------------------------------------------------
# What we just accomplished
# ---------------------------------------------------------------
print("""
============================================================
WE TRAINED A NEURAL NETWORK WITHOUT AUTOGRAD.
============================================================

Every gradient that updated every parameter in this network was
derived by hand from the chain rule. Cross-entropy through softmax,
matmul backward, BatchNorm's seven-step backward, embedding lookup's
index-accumulation -- all manual.

You now know what loss.backward() is doing under the hood. PyTorch's
autograd is a graph-walking machine that knows the local backward
rule for every primitive op. We just walked the graph ourselves.

This is the floor of every deep learning framework. JAX, TensorFlow,
MLX, PyTorch -- they all do exactly this. The differences are speed
(C++/CUDA vs. Python), API ergonomics, and which platforms they
target. The math is identical.

Why this lesson is the most-recommended of the series:
- You can now read research code that mixes manual gradients with
  autograd (common in custom ops, RL, meta-learning).
- You can debug NaN/Inf gradients by knowing exactly where they
  came from. "It came from the BN backward" -- you know which path.
- You appreciate why LayerNorm replaced BatchNorm in transformers:
  LN normalizes per-example, so its backward is simpler AND it doesn't
  need running stats -- it uses the same formula at training and
  inference. Big win.
- You're prepared for the next-level work: writing CUDA kernels,
  adding ops to JAX, contributing to PyTorch source.

The mountain is conquered. Onward to WaveNet (lesson 06).
""")
